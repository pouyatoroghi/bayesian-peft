import numpy  # needed (don't change it)
import importlib
import os
import socket
import sys
from ipdb import iex

from tempfile import TemporaryDirectory

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
sys.path.append(project_path + "/datasets")
sys.path.append(project_path + "/models")
sys.path.append(project_path + "/main")

import datetime
import uuid
from argparse import ArgumentParser

import setproctitle
import torch
from utils.args import add_management_args, add_experiment_args
from utils import create_if_not_exists

# from utils.continual_training import train as ctrain
from run import *

from accelerate.utils import set_seed
from accelerate import Accelerator

try:
    import wandb
except ImportError:
    wandb = None

from huggingface_hub import hf_hub_download
import torch
import numpy as np
from huggingface_hub import HfApi, upload_file
from collections import OrderedDict
import os

def are_models_identical_torch(model1, model2):
    """
    Check if two PyTorch models are absolutely identical in every aspect.
    
    Args:
        model1 (torch.nn.Module): First model to compare
        model2 (torch.nn.Module): Second model to compare
    
    Returns:
        bool: True if models are identical, False otherwise
    """
    # Check if models are the same class
    if model1.__class__ != model2.__class__:
        return False
    
    # Check model architecture by comparing state_dict keys
    if model1.state_dict().keys() != model2.state_dict().keys():
        return False
    
    # Check all parameters are exactly equal
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            return False
        if not torch.equal(param1, param2):
            return False
    
    # Check buffers (like running mean/variance in BatchNorm)
    for (name1, buf1), (name2, buf2) in zip(model1.named_buffers(), model2.named_buffers()):
        if name1 != name2:
            return False
        if not torch.equal(buf1, buf2):
            return False
    
    # Check training mode
    if model1.training != model2.training:
        return False
    
    return True

def get_model_layers_detailed(model):
    """
    Get detailed information about all layers in the model
    Returns a dictionary with layer names and their details
    """
    layers_info = {}
    
    for name, module in model.named_modules():
        # Skip empty modules (like top-level)
        if not list(module.named_children()):
            layers_info[name] = {
                'type': str(type(module)),
                'parameters': {n: p.shape for n, p in module.named_parameters()},
                'trainable': any(p.requires_grad for p in module.parameters())
            }
    
    return layers_info

def upload_lora_to_hub(model, repo_name, hf_token=None, filename="lora_weights.bin"):
    """
    Upload ONLY LoRA weights to HF Hub (consistent with loading method)
    
    Args:
        model: Your initialized model with LoRA
        repo_name: Repository name (e.g., "Uncertainty_BLOB")
        hf_token: Your HuggingFace token
        filename: Consistent filename for loading later
    """
    api = HfApi(token=hf_token)
    repo_id = f"Pouyatr/{repo_name}"
    
    # 1. Extract LoRA weights
    lora_weights = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only trainable (LoRA) params
            lora_weights[name] = param.data.clone().cpu()  # Ensure CPU tensor
    
    # 2. Create temporary file
    temp_path = f"temp_{filename}"
    torch.save(lora_weights, temp_path)
    
    # 3. Upload with consistent naming
    api.upload_file(
        path_or_fileobj=temp_path,
        path_in_repo=filename,  # Same filename used in loading
        repo_id=repo_id,
        token=hf_token,
        repo_type="model",
        commit_message="Upload LoRA weights"
    )
    
    # Cleanup
    os.remove(temp_path)
    print(f"✅ LoRA weights uploaded to: https://huggingface.co/{repo_id}")

def load_lora_from_hub(model, repo_name, args, accelerator, hf_token=None, filename="lora_weights.bin"):
    """
    Load LoRA weights into model from HF Hub (consistent with upload)
    
    Args:
        model: Pre-initialized model (with LoRA architecture)
        repo_name: Repository name (e.g., "Uncertainty_BLOB")
        hf_token: Your HuggingFace token
        filename: Must match upload filename
    Returns:
        Model with loaded LoRA weights
    """
    repo_id = f"Pouyatr/{repo_name}"
    
    # 1. Download with same filename used in upload
    try:
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=hf_token,
            library_name="lora-loader"
        )
    except Exception as e:
        raise ValueError(f"Failed to download {filename} from {repo_id}: {str(e)}")
    
    # 2. Load weights with device handling
    device = next(model.parameters()).device  # Preserve original device
    lora_weights = torch.load(weights_path, map_location=device)
    
    # 3. Filter for existing parameters only
    model_state = model.state_dict()
    updates = {}
    for name, param in lora_weights.items():
        if name in model_state:
            updates[name] = param
        elif f"model.{name}" in model_state:  # Handle common prefix issue
            updates[f"model.{name}"] = param
        else:
            print(f"⚠️ Skipping unused key: {name}")
    
    # 4. Load with strict=False to ignore non-LoRA params
    model.load_state_dict(updates, strict=False)
    return model

def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)

def parse_args():
    parser = ArgumentParser(description="Bayesian LoRA", allow_abbrev=False)
    add_management_args(parser)
    add_experiment_args(parser)
    args = parser.parse_known_args()[0]

    # add model-specific arguments
    mod = importlib.import_module("modelwrappers." + args.modelwrapper)
    get_parser = getattr(mod, "get_parser")
    parser = get_parser()  # the real parsing happens.
    args = parser.parse_args()

    # set random seed
    if args.seed is not None:
        set_seed(args.seed)

    return args

# @iex
def main(args=None):
    lecun_fix()
    if args is None:
        args = parse_args()

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    accelerator = Accelerator()

    dataset = get_dataset(args.dataset_type, accelerator, args)
    dataset.get_loaders()
    args.outdim = dataset.num_labels
    args.num_samples = dataset.num_samples

    # set job name
    setproctitle.setproctitle("{}_{}_BLoB-lora".format(args.model, args.dataset))

    # train the model

    wandb_logger = None
    if accelerator.is_local_main_process:
        print(args)
        if not args.nowand:
            assert (
                wandb is not None
            ), "Wandb not installed, please install it or run without wandb"
            if not args.wandb_name:
                wandb_logger = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    config=vars(args),
                )
            else:
                wandb_logger = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_name,
                    config=vars(args),
                )
        print(file=sys.stderr)

    print(f"Args: {args}")
    print(args.load_in_8bit)
    
    model = get_model(args, accelerator)
    modelwrapper = get_modelwrapper(args.modelwrapper)
    model.model = modelwrapper(
        model.model, model.peft_config, args, accelerator, adapter_name="default"
    )
    model.model.print_trainable_parameters()
    model.model.prepare_for_fit_evaluate(dataset, wandb_logger)    
    try:
        # Inference mode (load from Hub)
        hub_repo = f"{args.modelwrapper}_{args.model.split('/')[1]}_{args.dataset}_{args.max_train_steps}"
        assert hub_repo is not None, "hub_repo must be provided for inference"
        model.model = load_lora_from_hub(model.model, hub_repo, args, accelerator, hf_token=args.hf_token, filename="lora_weights.bin")
        model.model.evaluate(model.model.test_loader, model.model.val_loader)
    except:
        # Training mode
        model.model.fit_evaluate()
        upload_lora_to_hub(model.model, f"{args.modelwrapper}_{args.model.split('/')[1]}_{args.dataset}_{args.max_train_steps}", hf_token=args.hf_token, filename="lora_weights.bin")

    # print(f"Are they: {are_models_identical_torch(model.model, model1.model)}")
    # checkpointing the backbone model.
    if args.checkpoint:  # by default the checkpoints folder is checkpoints
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_folder = f"checkpoints/{args.modelwrapper}/{args.model}/{args.dataset}/{args.checkpoint_name}"
            create_if_not_exists(save_folder)
            model.model.base_model = accelerator.unwrap_model(model.model.base_model)
            model.model.save_pretrained(save_folder, save_function=accelerator.save)
            print("Model saved to:", save_folder)

    if not args.nowand:
        if accelerator.is_local_main_process:
            wandb_logger.finish()


if __name__ == "__main__":
    main()
