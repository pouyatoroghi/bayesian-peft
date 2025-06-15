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


def upload_model_to_hub(model, repo_name, hf_token):
    """
    Uploads the model to your existing repo: Pouyatr/Uncertainty_BLOB
    
    Args:
        model: Your trained model (BLoB + LoRA)
        hf_token: Your Hugging Face token (or use args.hf_token)
    """
    from huggingface_hub import HfApi, upload_folder
    import os
    import torch
    
    api = HfApi(token=hf_token)

    # Create a temporary directory
    with TemporaryDirectory() as tmp_dir:
        # 1. Save the base model
        model.model.base_model.save_pretrained(tmp_dir)
        
        # 2. Save BLoB-specific files
        blob_state = {
            'blobconfig': model.model.blobconfig,
            'args': model.model.args,
            'lora_A_rho': {
                name: param 
                for name, param in model.model.named_parameters()
                if 'lora_A_rho' in name
            },
        }
        torch.save(blob_state, os.path.join(tmp_dir, f'{repo_name}.bin'))

        # 3. Upload everything
        upload_folder(
            folder_path=tmp_dir,
            repo_id="Pouyatr/Uncertainty_BLOB",
            repo_type="model",
            token=hf_token,
            commit_message="Upload BLoB model with LoRA weights"
        )

    print(f"✅ Model uploaded to: https://huggingface.co/Pouyatr/Uncertainty_BLOB")


def load_from_hub_and_replace_lora(model, repo_name, args, accelerator):
    """
    Downloads BLoB weights from Hugging Face Hub and injects them into the model.

    Args:
        model: The target model to modify. This model should already be initialized
               with the correct architecture to receive the LoRA weights.
        repo_name (str): Hugging Face repository name (e.g., "username/repo-name").
        args: Command line arguments containing hf_token if needed.
        accelerator: For device placement (e.g., from Hugging Face Accelerate).

    Returns:
        The modified model with updated LoRA weights.
    """
    print(f"Attempting to load model from Hugging Face Hub: {repo_name}")

    # Authenticate if token is provided
    # The `login` function sets the token globally for huggingface_hub operations.
    if getattr(args, 'hf_token', None):
        print("Logging in with provided HF token.")
        login(token=args.hf_token)
    elif os.getenv('HF_TOKEN'):
        print("Logging in with HF token from environment variable.")
        login(token=os.getenv('HF_TOKEN'))
    else:
        print("No Hugging Face token found. Attempting to download public repository.")

    try:
        # Download model files
        # `snapshot_download` fetches the entire repository snapshot.
        download_kwargs = {
            'repo_id': repo_name,
            # 'allow_patterns': ['blob_state.bin', '*.json'], # Consider downloading all if base model files are needed
            'local_files_only': False, # Always try to download from remote first
        }

        # Add token if provided. It's good practice to pass it explicitly even if logged in.
        if getattr(args, 'hf_token', None):
            download_kwargs['token'] = args.hf_token

        model_dir = snapshot_download(**download_kwargs)
        print(f"Repository downloaded to: {model_dir}")

        # Load BLoB state with proper device handling
        # Determine the device for loading the tensor.
        # If accelerator is available, use its device; otherwise, default to 'cpu' or 'cuda'
        device = accelerator.device if accelerator and hasattr(accelerator, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading BLoB state to device: {device}")

        blob_path = os.path.join(model_dir, 'blob_state.bin')
        if not os.path.exists(blob_path):
            raise FileNotFoundError(f"blob_state.bin not found at {blob_path}. "
                                    "Ensure it was saved correctly during upload.")

        blob_state = torch.load(blob_path, map_location=device)
        print("BLoB state loaded.")

        # Replace lora_A_rho in the existing model
        # Iterate through the named parameters of the current model and update the LoRA weights.
        model_params = dict(model.model.named_parameters())
        updated_params_count = 0
        for name, param_from_blob in blob_state['lora_A_rho'].items():
            if name in model_params:
                # Ensure dimensions match before copying
                if model_params[name].data.shape == param_from_blob.data.shape:
                    model_params[name].data.copy_(param_from_blob.data)
                    updated_params_count += 1
                else:
                    print(f"Warning: Shape mismatch for parameter {name}. "
                          f"Model shape: {model_params[name].data.shape}, "
                          f"Loaded shape: {param_from_blob.data.shape}. Skipping update.")
            else:
                print(f"Warning: Parameter {name} not found in current model. Skipping.")

        if updated_params_count > 0:
            print(f"Successfully updated {updated_params_count} 'lora_A_rho' parameters.")
        else:
            print("No 'lora_A_rho' parameters were updated. Check parameter names and structure.")

        print(f"Model successfully loaded and LoRA weights replaced from: https://huggingface.co/{repo_name}")
        return model

    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise # Re-raise to stop execution if essential file is missing
    except Exception as e:
        print(f"Error loading model from Hub: {str(e)}")
        if "401" in str(e):
            print("Authentication failed - you may need to provide a valid --hf-token or ensure it has access to private repos.")
        elif "404" in str(e):
            print("Repository not found - check the repo_name. It must exactly match the name used for upload.")
        elif "Cannot load `model.safetensors`" in str(e) or "missing keys" in str(e):
             print("Model architecture mismatch or corrupt file. Ensure the base model for loading matches the uploaded one.")
        raise # Re-raise the exception after providing more context


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
    # model.model.print_trainable_parameters()
    # model.model.prepare_for_fit_evaluate(dataset, wandb_logger)
    # model.model.fit_evaluate()
    
    try:
        # Inference mode (load from Hub)
        hub_repo = f"{args.modelwrapper}_{args.model.split('/')[1]}_{args.dataset}_{args.max_train_steps}"
        assert hub_repo is not None, "hub_repo must be provided for inference"
        model = load_from_hub_and_replace_lora(model, hub_repo, args, accelerator)
    except:
        # Training mode
        model.model.print_trainable_parameters()
        model.model.prepare_for_fit_evaluate(dataset, wandb_logger)
        model.model.fit_evaluate()
        upload_model_to_hub(model, f"{args.modelwrapper}_{args.model.split('/')[1]}_{args.dataset}_{args.max_train_steps}", args.hf_token)
    
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
