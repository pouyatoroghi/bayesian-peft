import numpy  # needed (don't change it)
import importlib
import os
import socket
import sys
from ipdb import iex

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

def upload_model_to_hub(model, repo_name):
    """
    Uploads the wrapped model (BLoB + LoRA) to Hugging Face Hub.
    
    Args:
        model: Your full model (with .model being the BLoB-wrapped model)
        tokenizer: The associated tokenizer
        repo_name: Name of the repository (e.g., "blob-qwen-7b")
    """
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp_dir:
        # Save the base model (inside the wrapper)
        model.model.base_model.save_pretrained(tmp_dir)  # Now accessing .model.base_model
        
        if tokenizer is not None:
            tokenizer.save_pretrained(tmp_dir)

        # Save BLoB-specific parameters (e.g., lora_A_rho)
        blob_state = {
            'blobconfig': model.model.blobconfig,  # Access via .model
            'args': model.model.args,
            'lora_A_rho': {
                name: param 
                for name, param in model.model.named_parameters()  # Note: .model here
                if 'lora_A_rho' in name
            },
        }
        torch.save(blob_state, os.path.join(tmp_dir, 'blob_state.bin'))

        # Push to Hub
        repo_url = HfApi().create_repo(
            repo_id=repo_name,
            exist_ok=True,
            private=False,
        )
        repo = Repository(tmp_dir, clone_from=repo_url)
        repo.push_to_hub(commit_message="Upload BLoB-wrapped Qwen model")

    print(f"Model uploaded to: https://huggingface.co/{repo_name}")

def load_from_hub_and_replace_lora(model, repo_name, args, accelerator):
    """
    Downloads BLoB weights from Hugging Face Hub and injects them into the model.
    """
    from huggingface_hub import snapshot_download
    
    # 1. Download model files
    model_dir = snapshot_download(repo_id=repo_name)
    
    # 2. Load BLoB state
    blob_state = torch.load(os.path.join(model_dir, 'blob_state.bin'), map_location="cuda")
    
    # 3. Replace lora_A_rho in the existing model
    for name, param in blob_state['lora_A_rho'].items():
        model_param = dict(model.model.named_parameters())[name]
        model_param.data.copy_(param.data)

    print(f"Model loaded from: https://huggingface.co/{repo_name}")
    
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
    # model.model.print_trainable_parameters()
    # model.model.prepare_for_fit_evaluate(dataset, wandb_logger)
    # model.model.fit_evaluate()
    
    try:
        # Inference mode (load from Hub)
        hub_repo = f"{args.modelwrapper}_{args.model}_{args.dataset}_{args.max_train_steps}"
        assert hub_repo is not None, "hub_repo must be provided for inference"
        model = load_from_hub_and_replace_lora(model, hub_repo, args, accelerator)
    except:
        # Training mode
        model.model.print_trainable_parameters()
        model.model.prepare_for_fit_evaluate(dataset, wandb_logger)
        model.model.fit_evaluate()
        upload_model_to_hub(model, f"{args.modelwrapper}_{args.model.split("/")}_{args.dataset}_{args.max_train_steps}")
    
    # checkpointing the backbone model.
    if args.checkpoint:  # by default the checkpoints folder is checkpoints
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_folder = f"checkpoints/{args.modelwrapper}/{args.model.split("/")}/{args.dataset}/{args.checkpoint_name}"
            create_if_not_exists(save_folder)
            model.model.base_model = accelerator.unwrap_model(model.model.base_model)
            model.model.save_pretrained(save_folder, save_function=accelerator.save)
            print("Model saved to:", save_folder)

    if not args.nowand:
        if accelerator.is_local_main_process:
            wandb_logger.finish()


if __name__ == "__main__":
    main()
