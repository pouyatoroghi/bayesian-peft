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


# def upload_model_to_hub(model, repo_name, hf_token):
#     """
#     Uploads the model to your existing repo: Pouyatr/Uncertainty_BLOB
    
#     Args:
#         model: Your trained model (BLoB + LoRA)
#         hf_token: Your Hugging Face token (or use args.hf_token)
#     """
#     from huggingface_hub import HfApi, upload_folder
#     import os
#     import torch
    
#     api = HfApi(token=hf_token)

#     # Create a temporary directory
#     with TemporaryDirectory() as tmp_dir:
#         # 1. Save the base model
#         model.model.base_model.save_pretrained(tmp_dir)
        
#         # 2. Save BLoB-specific files
#         blob_state = {
#             'blobconfig': model.model.blobconfig,
#             'args': model.model.args,
#             'lora_A_rho': {
#                 name: param 
#                 for name, param in model.model.named_parameters()
#                 if 'lora_A_rho' in name
#             },
#         }
#         torch.save(blob_state, os.path.join(tmp_dir, 'blob_state.bin'))

#         # 3. Upload everything
#         upload_folder(
#             folder_path=tmp_dir,
#             repo_id=f"Pouyatr/{repo_name}",
#             repo_type="model",
#             token=hf_token,
#             commit_message="Upload BLoB model with LoRA weights"
#         )

#     print(f"✅ Model uploaded to: https://huggingface.co/Pouyatr/{repo_name}")

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

# To save the full state dict (all parameters, not just LoRA):
def save_full_state_dict(model, path='full_model.pth'):
    torch.save(model.state_dict(), path)

# def upload_model_to_hub(model, repo_name, hf_token):
#     """
#     Uploads the model to your existing repo: Pouyatr/Uncertainty_BLOB
    
#     Args:
#         model: Your trained model (BLoB + LoRA)
#         repo_name: Name of the repository to create/upload to
#         hf_token: Your Hugging Face token (or use args.hf_token)
#     """
#     from huggingface_hub import HfApi, upload_folder, create_repo
#     import os
#     import torch
#     from tempfile import TemporaryDirectory
    
#     api = HfApi(token=hf_token)
#     repo_id = f"Pouyatr/{repo_name}"

#     # Create the repository if it doesn't exist
#     try:
#         create_repo(
#             repo_id=repo_id,
#             token=hf_token,
#             exist_ok=True,  # Won't raise error if repo exists
#             repo_type="model"
#         )
#     except Exception as e:
#         print(f"⚠️ Could not create repository: {e}")
#         raise

#     # Create a temporary directory
#     with TemporaryDirectory() as tmp_dir:
#         # 1. Save the base model and adapter
#         model.model.save_pretrained(tmp_dir)
        
#         # 2. Save BLoB-specific files
#         blob_state = {
#             'blobconfig': model.model.blobconfig,
#             'args': model.model.args,
#             'lora_A_rho': {
#                 name: param 
#                 for name, param in model.model.named_parameters()
#                 if 'lora_A_rho' in name
#             },
#         }
#         torch.save(blob_state, os.path.join(tmp_dir, 'blob_state.bin'))

#         # 3. Upload everything
#         upload_folder(
#             folder_path=tmp_dir,
#             repo_id=repo_id,
#             repo_type="model",
#             token=hf_token,
#             commit_message="Upload BLoB model with LoRA weights"
#         )

#     print(f"✅ Model uploaded to: https://huggingface.co/{repo_id}")

# def load_from_hub_and_replace_lora(model, repo_name, args, accelerator):
#     """
#     Downloads and loads all model components from Hugging Face Hub.
    
#     Args:
#         model: Target model to modify
#         repo_name: Repository name (e.g., "Pouyatr/Uncertainty_BLOB")
#         args: Command line arguments
#         accelerator: For device placement
#     """
#     from huggingface_hub import snapshot_download, login
#     from peft import PeftModel
#     import torch
#     import os
#     from peft import LoraConfig

#     # Authenticate if needed
#     if getattr(args, 'hf_token', None):
#         login(token=args.hf_token)

#     try:
#         # Download all repository files
#         model_dir = snapshot_download(
#             repo_id=repo_name,
#             allow_patterns=["*.bin", "*.safetensors", "*.json"],
#             token=getattr(args, 'hf_token', None)
#         )

#         device = accelerator.device if accelerator else 'cuda'

#         # 1. Load adapter config first
#         if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
#             peft_config = LoraConfig.from_pretrained(model_dir)
#             print("✅ Loaded adapter configuration")
#             print(peft_config)

#         from peft import (get_peft_model, LoraConfig, PeftModel, PeftConfig)

#         target_modules = ["lm_head", "v_proj", "q_proj"]
        
#         peft_config = LoraConfig(
#             task_type="CAUSAL_LM",
#             inference_mode=True,
#             r=args.lora_r,
#             lora_alpha=args.lora_alpha,
#             lora_dropout=args.lora_dropout,
#             target_modules=target_modules
#         )

#         # print(model)              # Shows the outer container
#         # print(model.model)        # Shows the BLoB-wrapped model
#         # print(type(model.model))  # Should be your BLoB wrapper class

#         linf = get_model_layers_detailed(model.model)
#         for key, value in linf.items():
#             print(f"\033[31m{key}: {value}\033[0m")
        
#         # 2. Load adapter weights (LoRA)
#         if os.path.exists(os.path.join(model_dir, "adapter_model.safetensors")):
#             model.model = PeftModel.from_pretrained(
#                 model,
#                 model_dir,
#                 config=peft_config,
#                 device_map={"": device},
#                 is_trainable=True
#             )
#             print("✅ Successfully loaded adapter weights")

#         # 3. Load BLoB state
#         if os.path.exists(os.path.join(model_dir, "blob_state.bin")):
#             blob_state = torch.load(
#                 os.path.join(model_dir, "blob_state.bin"),
#                 map_location=device
#             )
            
#             # Update model parameters
#             model_params = dict(model.model.named_parameters())
#             for name, param in blob_state['lora_A_rho'].items():
#                 if name in model_params:
#                     model_params[name].data.copy_(param.data)
#             print("✅ Successfully loaded BLoB state")

#         print(f"🎉 Successfully loaded all components from {repo_name}")
#         return model

#     except Exception as e:
#         print(f"❌ Error loading model: {str(e)}")
#         raise

from huggingface_hub import HfApi, upload_file
import torch
from collections import OrderedDict
import os

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


from huggingface_hub import hf_hub_download

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


# def load_from_hub_and_replace_lora(model, repo_name, args, accelerator):
#     """
#     Downloads and loads all model components from Hugging Face Hub.
    
#     Args:
#         model: Target model to modify
#         repo_name: Repository name (e.g., "Pouyatr/Uncertainty_BLOB")
#         args: Command line arguments
#         accelerator: For device placement
#     """
#     from huggingface_hub import snapshot_download, login
#     from peft import PeftModel
#     import torch
#     import os

#     # Authenticate if needed
#     if getattr(args, 'hf_token', None):
#         login(token=args.hf_token)

#     # try:
#     if True:
#         # Download all repository files
#         model_dir = snapshot_download(
#             repo_id=repo_name,
#             allow_patterns=["*.bin", "*.safetensors", "*.json"],
#             token=getattr(args, 'hf_token', None)
#         )

#         device = accelerator.device if accelerator else 'cuda'

#         # 1. Load adapter weights (LoRA)
#         if os.path.exists(os.path.join(model_dir, "adapter_model.safetensors")):
#             model = PeftModel.from_pretrained(
#                 model,
#                 model_dir,
#                 device_map={"": device},
#                 is_trainable=True
#             )
#             print("✅ Successfully loaded adapter weights")

#         # 2. Load BLoB state with proper safety settings
#         if os.path.exists(os.path.join(model_dir, "blob_state.bin")):
#             import torch.serialization
#             from modelwrappers.blob import BLoBConfig  # Import your custom config class
            
#             # Allow your custom BLoBConfig class to be loaded safely
#             with torch.serialization.safe_globals([BLoBConfig]):
#                 blob_state = torch.load(
#                     os.path.join(model_dir, "blob_state.bin"),
#                     map_location=device,
#                     weights_only=False  # Required for custom classes
#                 )
            
#             # Update model parameters
#             model_params = dict(model.model.named_parameters())
#             for name, param in blob_state['lora_A_rho'].items():
#                 if name in model_params:
#                     model_params[name].data.copy_(param.data)
#             print("✅ Successfully loaded BLoB state")

#         # 3. Load any additional configs
#         if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
#             # Handle any additional configuration loading here
#             print("✅ Loaded adapter configuration")

#         print(f"🎉 Successfully loaded all components from {repo_name}")
#         return model

    # except Exception as e:
    #     print(f"❌ Error loading model: {str(e)}")
    #     if "404" in str(e):
    #         print("Repository or files not found - check the name and permissions")
    #     return model  # Return original model on failure

# def upload_model_to_hub(model, repo_name, hf_token):
#     """
#     Uploads the wrapped model (BLoB + LoRA) to Hugging Face Hub.
#     """
#     from tempfile import TemporaryDirectory
#     from huggingface_hub import HfApi, Repository

#     api = HfApi(token=hf_token)

#     # Create repo first
#     api.create_repo(
#         repo_id=repo_name,
#         exist_ok=True,
#         private=False,
#     )

#     with TemporaryDirectory() as tmp_dir:
#         # Save the base model (inside the wrapper)
#         model.model.base_model.save_pretrained(tmp_dir)

#         # Save BLoB-specific parameters
#         blob_state = {
#             'blobconfig': model.model.blobconfig,
#             'args': model.model.args,
#             'lora_A_rho': {
#                 name: param 
#                 for name, param in model.model.named_parameters()
#                 if 'lora_A_rho' in name
#             },
#         }
#         torch.save(blob_state, os.path.join(tmp_dir, 'blob_state.bin'))

#         # Initialize a new repository in the temp dir
#         repo = Repository(tmp_dir, clone_from=repo_name, token=hf_token, skip_lfs_files=True)
        
#         # Add all files and commit
#         repo.git_add(auto_lfs_track=True)
#         repo.git_commit("Upload BLoB-wrapped Qwen model")
#         repo.git_push()

#     print(f"Model uploaded to: https://huggingface.co/{repo_name}")


# def load_from_hub_and_replace_lora(model, repo_name, args, accelerator):
#     """
#     Downloads BLoB weights from Hugging Face Hub and injects them into the model.
    
#     Args:
#         model: The target model to modify
#         repo_name: Hugging Face repository name (e.g., "username/repo-name")
#         args: Command line arguments containing hf_token if needed
#         accelerator: For device placement
    
#     Returns:
#         The modified model with updated LoRA weights
#     """
#     from huggingface_hub import snapshot_download, login
    
#     # Authenticate if token is provided
#     if getattr(args, 'hf_token', None):
#         login(token=args.hf_token)
#     elif os.getenv('HF_TOKEN'):
#         login(token=os.getenv('HF_TOKEN'))
    
#     try:
#         # Download model files (private repos will need auth)
#         download_kwargs = {
#             'repo_id': repo_name,
#             'allow_patterns': ['blob_state.bin', '*.json'],
#             'local_files_only': False
#         }
        
#         # Add token if provided
#         if getattr(args, 'hf_token', None):
#             download_kwargs['token'] = args.hf_token
            
#         model_dir = snapshot_download(**download_kwargs)
        
#         # Load BLoB state with proper device handling
#         device = accelerator.device if accelerator else 'cuda'
#         blob_path = os.path.join(model_dir, 'blob_state.bin')
#         blob_state = torch.load(blob_path, map_location=device)
        
#         # Replace lora_A_rho in the existing model
#         model_params = dict(model.model.named_parameters())
#         for name, param in blob_state['lora_A_rho'].items():
#             if name in model_params:
#                 model_params[name].data.copy_(param.data)
#             else:
#                 print(f"Warning: Parameter {name} not found in model")
        
#         print(f"Successfully loaded model from: https://huggingface.co/{repo_name}")
#         return model
        
#     except Exception as e:
#         print(f"Error loading model from Hub: {str(e)}")
#         if "401" in str(e):
#             print("Authentication failed - you may need to provide a valid --hf-token")
#         elif "404" in str(e):
#             print("Repository not found - check the repo_name")
#         raise

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
    # linf = get_model_layers_detailed(model.model)
    # for key, value in linf.items():
    #     print(f"{key}: {value}")
    
    
    # print(1, model)              # Shows the outer container
    # print(1, model.model)        # Shows the BLoB-wrapped model
    # print(1, type(model.model))  # Should be your BLoB wrapper class
    model.model.print_trainable_parameters()
    model.model.prepare_for_fit_evaluate(dataset, wandb_logger)
    # model.model.fit_evaluate()
    
    try:
        # Inference mode (load from Hub)
        hub_repo = f"{args.modelwrapper}_{args.model.split('/')[1]}_{args.dataset}_{args.max_train_steps}"
        assert hub_repo is not None, "hub_repo must be provided for inference"
        # model = load_from_hub_and_replace_lora(model, hub_repo, args, accelerator)
        model.model = load_lora_from_hub(model.model, hub_repo, args, accelerator, hf_token=args.hf_token, filename="lora_weights.bin")
    except:
        # Training mode
        # model.model.print_trainable_parameters()
        # model.model.prepare_for_fit_evaluate(dataset, wandb_logger)
        model.model.fit_evaluate()
        # upload_model_to_hub(model, f"{args.modelwrapper}_{args.model.split('/')[1]}_{args.dataset}_{args.max_train_steps}", args.hf_token)
        upload_lora_to_hub(model.model, f"{args.modelwrapper}_{args.model.split('/')[1]}_{args.dataset}_{args.max_train_steps}", hf_token=args.hf_token, filename="lora_weights.bin")
    
    # Inference mode (load from Hub)
    hub_repo = f"{args.modelwrapper}_{args.model.split('/')[1]}_{args.dataset}_{args.max_train_steps}"
    assert hub_repo is not None, "hub_repo must be provided for inference"
    model.model = load_lora_from_hub(model.model, hub_repo, args, accelerator, hf_token=args.hf_token, filename="lora_weights.bin")
    model.model.evaluate(model.model.test_loader, model.model.val_loader)
    
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
