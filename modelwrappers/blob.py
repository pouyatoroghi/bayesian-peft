import torch
import torch.nn as nn
from torch.optim import SGD
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union
import math
from tqdm import tqdm
from dataset.utils.dsets import BOSSDataset
import logging

from .wrapperbase import WrapperBase, get_linear_schedule_with_warmup
from utils.args import add_management_args, add_experiment_args, ArgumentParser
from run.evaluation import *

from transformers import PreTrainedModel

from peft.config import PeftConfig
from peft.tuners.lora import LoraLayer, Linear
from peft.tuners.lora.bnb import Linear8bitLt
from transformers import AutoTokenizer

## Model Specific Argument Parsing
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Bayesian By Backprop, BLoB.")
    add_management_args(parser)
    add_experiment_args(parser)
    # BLoB-specific arguments.
    parser.add_argument("--bayes-train-n-samples", type=int, default=1)
    parser.add_argument(
        "--bayes-eval-n-samples",
        type=int,
        default=1,
        help="Number of samples to use for evaluation during training.",
    )
    parser.add_argument(
        "--bayes-eval-n-samples-final",
        type=int,
        default=10,
        help="Number of samples to use for evaluation.",
    )

    parser.add_argument("--bayes-eps", type=float, default=0.05)
    parser.add_argument("--bayes-gamma", type=float, default=8)
    parser.add_argument("--bayes-kllr", type=float, default=0.02)
    parser.add_argument("--bayes-beta", type=float, default=0.2)
    parser.add_argument(
        "--bayes-inference-notsample",
        action="store_true",
        help="Whether to sample during inference.",
    )
    parser.add_argument(
        "--bayes-klreweighting", action="store_true", help="Whether to use reweighting."
    )
    parser.add_argument('--bayes-datasetrescaling', action='store_true',
                        help='Whether to use datasetrescaling.')

    return parser


@dataclass
class BLoBConfig:
    bayes_eps: float = field(metadata={"help": "Bayes epsilon"})
    bayes_gamma: float = field(metadata={"help": "Bayes gamma"})
    bayes_beta: float = field(metadata={"help": "Bayes beta"})


def blob_linear_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
    previous_dtype = x.dtype

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)
            result += lora_B(lora_A(dropout(x))) * scaling

    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A.keys():
            continue
        lora_A = self.lora_A[active_adapter]
        if self.blobsample:
            if self.bayes_eps < 0:
                A_sigma = torch.log1p(torch.exp(self.lora_A_rho[active_adapter]))
            else:
                A_sigma = self.lora_A_rho[active_adapter] ** 2

            scaling = self.scaling[active_adapter]
            dropout = self.lora_dropout[active_adapter]

            x = x.to(lora_A.weight.dtype)
            if x.dim() == 2:
                r_A = (
                    torch.ones(
                        (x.size(0), self.in_features), device=x.device, dtype=x.dtype
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
                s_A = (
                    torch.ones(
                        (x.size(0), self.r[active_adapter]),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
            else:
                r_A = (
                    torch.ones(
                        (x.size(0), x.size(1), self.in_features),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
                s_A = (
                    torch.ones(
                        (x.size(0), x.size(1), self.r[active_adapter]),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )

            x = dropout(x)
            lora_noise_a = A_sigma * torch.randn_like(
                self.lora_A[active_adapter].weight
            )

            noise = (((x * r_A) @ lora_noise_a.transpose(0, 1)) * s_A) @ self.lora_B[
                active_adapter
            ].weight.transpose(0, 1)

            result += noise * scaling

        result = result.to(previous_dtype)

    return result


def blob_8bitlinear_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                compute_dtype = lora_A.weight.dtype
                if x.dtype != compute_dtype:
                    x = x.to(compute_dtype)
            output = lora_B(lora_A(dropout(x)))
            if requires_conversion:
                output = output.to(expected_dtype)
            output = output * scaling
            result += output
    if self.blobsample:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            if self.bayes_eps < 0:
                A_sigma = torch.log1p(torch.exp(self.lora_A_rho[active_adapter]))
            else:
                A_sigma = self.lora_A_rho[active_adapter] ** 2
            scaling = self.scaling[active_adapter]
            dropout = self.lora_dropout[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                compute_dtype = lora_A.weight.dtype
                if x.dtype != compute_dtype:
                    x = x.to(compute_dtype)

            if x.dim() == 2:
                r_A = (
                    torch.ones(
                        (x.size(0), self.in_features), device=x.device, dtype=x.dtype
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
                s_A = (
                    torch.ones(
                        (x.size(0), self.r[active_adapter]),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
            else:
                r_A = (
                    torch.ones(
                        (x.size(0), x.size(1), self.in_features),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
                s_A = (
                    torch.ones(
                        (x.size(0), x.size(1), self.r[active_adapter]),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )

            x = dropout(x)
            lora_noise_a = A_sigma * torch.randn_like(
                self.lora_A[active_adapter].weight
            )

            noise = (((x * r_A) @ lora_noise_a.transpose(0, 1)) * s_A) @ self.lora_B[
                active_adapter
            ].weight.transpose(0, 1)

            if requires_conversion:
                noise = noise.to(expected_dtype)

            result += noise * scaling

    return result


def div_posterior_prior(self) -> torch.Tensor:
    def kl_div_stable(mu_q, sigma_q, mu_p, sigma_p):
        eps = 1e-6
        kl = (
            math.log(sigma_p + eps)
            - torch.log(sigma_q.to(torch.float64) + eps)
            + (sigma_q.to(torch.float64) ** 2 + (mu_q.to(torch.float64) - mu_p) ** 2)
            / (2 * (sigma_p**2) + eps)
            - 0.5
        )
        return kl.sum()

    kl = 0
    for active_adapter in self.active_adapters:
        if self.bayes_eps < 0:
            sigma_weight = torch.log1p(torch.exp(self.lora_A_rho[active_adapter]))
        else:
            sigma_weight = self.lora_A_rho[active_adapter] ** 2
        kl += kl_div_stable(
            self.lora_A[active_adapter].weight, sigma_weight, 0, self.bayes_beta
        )
    return kl


def sample(self, status=True):
    if self.training is True and status is False:
        raise ValueError("blobsample should be set to True only during training.")
    self.blobsample = status


class BLoB(WrapperBase):
    """BLoB model."""

    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        args,
        accelerator,
        adapter_name: str = "default",
    ):
        super().__init__(model, peft_config, args, accelerator, adapter_name)

        self.blobconfig = BLoBConfig(
            bayes_eps=self.args.bayes_eps,
            bayes_gamma=self.args.bayes_gamma,
            bayes_beta=self.args.bayes_beta,
        )
        self.val_loader = None
        self._modify_lora_layers(self.base_model)
        if args.load_lora_path is not None:
            self.load_adapter(args.load_lora_path, adapter_name)

        self.i = 1  # for the KL re-weighting.
        self.ii = 1
        self.M = 0  # for the KL re-weighting.

        self.train_n_samples = self.args.bayes_train_n_samples
        self.eval_n_samples = self.args.bayes_eval_n_samples
        self.klreweighting = self.args.bayes_klreweighting

        if self.args.max_train_steps == 0:
            num_training_steps = (
                self.args.num_samples * self.args.n_epochs // self.args.batch_size
            )
        else:
            num_training_steps = self.args.max_train_steps
        warmup_steps = num_training_steps * self.args.warmup_ratio

        params = [param for name, param in self.named_parameters()]
        self.opt2 = SGD([{"params": params}], lr=args.bayes_kllr)
        self.scheduler2 = get_linear_schedule_with_warmup(
            self.opt2, warmup_steps, num_training_steps
        )

    def _modify_lora_layers(self, module):
        """
        Recursively go through the model and modify LoraLayer instances.
        """
        for name, child in module.named_children():
            if isinstance(child, LoraLayer) and isinstance(child, Linear):
                self._wrap_lora_layer(child)
                # modify existing methods
                setattr(
                    child,
                    "forward",
                    blob_linear_forward.__get__(child, child.__class__),
                )
                # add new methods
                setattr(
                    child,
                    "div_posterior_prior",
                    div_posterior_prior.__get__(child, child.__class__),
                )
                setattr(child, "sample", sample.__get__(child, child.__class__))
            if isinstance(child, LoraLayer) and isinstance(child, Linear8bitLt):
                self._wrap_lora_layer(child)
                # modify existing methods
                setattr(
                    child,
                    "forward",
                    blob_8bitlinear_forward.__get__(child, child.__class__),
                )
                # add new methods
                setattr(
                    child,
                    "div_posterior_prior",
                    div_posterior_prior.__get__(child, child.__class__),
                )
                setattr(child, "sample", sample.__get__(child, child.__class__))
            else:
                self._modify_lora_layers(child)

    def _wrap_lora_layer(self, lora_layer):
        lora_layer.lora_A_rho = nn.ParameterDict({})
        lora_layer.bayes_eps = self.blobconfig.bayes_eps
        lora_layer.bayes_gamma = self.blobconfig.bayes_gamma
        lora_layer.bayes_beta = self.blobconfig.bayes_beta
        lora_layer.blobsample = True

        for adapter_name in lora_layer._active_adapter:
            lora_layer.lora_A_rho[adapter_name] = nn.Parameter(
                lora_layer.lora_A[adapter_name].weight.new_zeros(
                    lora_layer.r[adapter_name], lora_layer.in_features
                )
            )

        if adapter_name in lora_layer.lora_A.keys():
            if lora_layer.bayes_eps < 0:
                nn.init.uniform_(
                    lora_layer.lora_A_rho[adapter_name],
                    lora_layer.bayes_eps - 1,
                    lora_layer.bayes_eps,
                )
            else:
                nn.init.uniform_(
                    lora_layer.lora_A_rho[adapter_name],
                    lora_layer.bayes_eps / math.sqrt(2),
                    lora_layer.bayes_eps,
                )

        return

    def div_posterior_prior(self, module):
        kl = 0
        for name, child in module.named_children():
            if isinstance(child, LoraLayer):
                kl_ = child.div_posterior_prior()
                # if not math.isnan(kl_):
                kl += kl_
            else:
                kl += self.div_posterior_prior(child)
        return kl

    def sample(self, module, status=True):
        """
        Set the sampling status of the model.
        """
        for name, child in module.named_children():
            if isinstance(child, LoraLayer):
                child.sample(status)
            else:
                self.sample(child, status)

    def forward_logits(self, batch, sample=True, n_samples=1, **kwargs) -> torch.Tensor:
        if self.args.dataset_type == "mcdataset":
            inputs, _, _ = batch
            if not sample:
                self.sample(self.base_model, False)
                output = self.base_model(**inputs)
                # print(f"Logits shape: {output.logits.shape}")
                logits = output.logits[:, -1, self.target_ids]
                self.sample(self.base_model, True)
                return logits.unsqueeze(1)
            else:
                logits_list = []
                for _ in range(n_samples):
                    output = self.base_model(**inputs)
                    logits = output.logits[:, -1, self.target_ids]
                    logits_list.append(logits)
                return torch.stack(logits_list, dim=1)
        else:
            if not sample:
                self.sample(self.base_model, False)
                res = self.base_model(**batch).logits
                self.sample(self.base_model, True)
                return res.unsqueeze(1)
            else:
                res = []
                for _ in range(n_samples):
                    res.append(self.base_model(**batch).logits)
                return torch.stack(res, dim=1)

    def fit(self, train_loader, test_loader, val_loader):
        print("FITTing started!")
        nll_losses = AverageMeter()
        kl_losses = AverageMeter()
        elbo_losses = AverageMeter()
        accs = AverageMeter()
        samples_seen = 0
        # with tqdm(
        #     total=len(train_loader),
        #     desc=f"Epoch {self.args.epoch+1}/{self.args.n_epochs}",
        #     leave=False,
        # ) as pbar:
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {self.args.epoch+1}/{self.args.n_epochs}", leave=True)
        for i, batch in enumerate(train_loader):
                if self.args.dataset_type == "mcdataset":
                    _, golds, _ = batch
                elif self.args.dataset_type == "bertds":
                    golds = batch["labels"]
                else:
                    raise NotImplementedError(
                        f"Dataset type {self.args.dataset_type} not implemented."
                    )
                logits = self.forward_logits(
                    batch, sample=True, n_samples=self.train_n_samples
                ).mean(1)
                output = torch.log_softmax(logits, dim=1)
                nll = self.loss(output, golds, reduction="mean")

                self.accelerator.backward(nll)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                kl_divs = []
                for _ in range(self.train_n_samples):
                    if hasattr(self.base_model, "module"):
                        kl_divs.append(self.div_posterior_prior(self.base_model.module))
                    else:
                        kl_divs.append(self.div_posterior_prior(self.base_model))
                kl = torch.mean(torch.stack(kl_divs), dim=0)

                if self.klreweighting:
                    if self.i % self.M == 0:
                        i = self.M
                    else:
                        i = self.i % self.M
                    self.pi = 2**i / (2 ** (self.M + 1) - 1)
                    self.i += 1
                else:
                    self.pi = 1 / self.M
                kl_div = kl * self.pi
                self.accelerator.backward(kl_div)
                self.opt2.step()
                self.opt2.zero_grad()
                self.scheduler2.step()

                acc = accuracy_topk(output.data, golds)

                loss, acc, nll_loss, kl = (
                    (kl + nll).detach().cpu().numpy(),
                    acc.item(),
                    nll.detach().cpu().numpy(),
                    kl_div.detach().cpu().numpy(),
                )

                if self.args.dataset_type == "mcdataset":
                    _, classes, _ = batch
                    references = self.accelerator.gather(classes)
                else:
                    references = self.accelerator.gather(batch["labels"])
                if self.accelerator.num_processes > 1:
                    if i == len(train_loader) - 1:
                        references = references[
                            : len(train_loader.dataset) - samples_seen
                        ]
                    else:
                        samples_seen += references.shape[0]
                len_batch = references.shape[0]
                kl_losses.update(kl, len_batch)
                nll_losses.update(nll_loss, len_batch)
                elbo_losses.update(loss, len_batch)
                accs.update(acc, len_batch)

                assert not math.isnan(nll_loss)
                assert not math.isnan(kl)
                if self.accelerator.is_local_main_process:
                    if self.wandb_logger is not None:
                        self.wandb_logger.log(
                            {
                                "train_acc": accs.avg,
                                "train_nll_loss": nll_losses.avg,
                                "kl_loss": kl_losses.avg,
                                "elbo_loss": elbo_losses.avg,
                                "lr": self.opt.param_groups[0]["lr"],
                                "pi": self.pi,
                            }
                        )

                self.step += self.accelerator.num_processes
                pbar.update(1)
                # if i == 10:
                #     break
                # if self.step >= self.args.eval_per_steps:
                #     self.step -= self.args.eval_per_steps
                #     self.evaluate(test_loader, val_loader)

    # def evaluate(self, test_loader, val_loader):
    #     print("EVALUATing started!")
    #     print("self.eval_n_samples:", self.eval_n_samples)
    #     self.eval()
    #     status = self.training
    #     nlls = AverageMeter()
    #     metric_kwargs = {"task": "multiclass", "num_classes": self.num_classes}
    #     acc_metric = Accuracy(**metric_kwargs).to(self.accelerator.device)
    #     ece_metric = CalibrationError(**metric_kwargs, n_bins=self.args.num_bins).to(
    #         self.accelerator.device
    #     )
    #     briers = AverageMeter()

    #     samples_seen = 0
    #     for step, batch in enumerate(eval_loader):
    #         with torch.no_grad() and torch.inference_mode():
    #             logits = self.forward_logits(
    #                 batch,
    #                 sample=not self.args.bayes_inference_notsample,
    #                 n_samples=self.eval_n_samples,
    #             ).detach()
    #             if self.args.dataset_type == "mcdataset":
    #                 _, labels, _ = batch
    #             else:
    #                 labels = batch["labels"]
    #             logits, labels = self.accelerator.gather([logits, labels])
    #             if self.accelerator.num_processes > 1:
    #                 if step == len(eval_loader) - 1:
    #                     labels = labels[: len(eval_loader.dataset) - samples_seen]
    #                     logits = logits[: len(eval_loader.dataset) - samples_seen]
    #                 else:
    #                     samples_seen += labels.shape[0]
    #             probs = torch.softmax(logits, dim=-1).mean(dim=1)
    #             std = torch.softmax(logits, dim=-1).std(dim=1).mean()

    #             # print(f"Probs: {probs}, {probs.shape}")
    #             # print(f"Labels: {labels}, {labels.shape}")
    #             # print(metric_kwargs)
    #             # idx, batch = next(enumerate(eval_loader))
    #             # inputs, _, _ = batch
    #             # self.sample(self.base_model, False)
    #             # output = self.base_model(**inputs)
    #             # print(f"Logits shape: {output.logits.shape}, {self.target_ids}")
    #             # # Get predicted token IDs (greedy decoding)
    #             # predicted_ids = torch.argmax(logits, dim=-1)
    #             # model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Replace with your desired Qwen variant
    #             # ttokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    #             # # Decode to text
    #             # generated_text = ttokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    #             # print(f"Text :!{generated_text}!")
                
    #             acc_metric(probs, labels)
    #             ece_metric(probs, labels)
    #             nll = self.loss(torch.log(probs), labels, reduction="mean")
    #             if torch.isnan(nll):
    #                 if self.accelerator.is_local_main_process:
    #                     print("nll:", nll)
    #                     print("probs:", probs)
    #                     print("logits:", logits)
    #                     exit()
    #             nlls.update(nll)

    #             brier = (
    #                 (probs - F.one_hot(labels, num_classes=logits.size(-1)))
    #                 .pow(2)
    #                 .sum(dim=-1)
    #                 .mean()
    #             )
    #             briers.update(brier)

    #     val_acc = acc_metric.compute().item()
    #     val_ece = ece_metric.compute().item()
    #     val_nll = nlls.avg
    #     val_brier = briers.avg
    #     self.train(status)

    #     if self.accelerator.is_local_main_process:
    #         if self.wandb_logger is not None:
    #             self.wandb_logger.log(
    #                 {
    #                     "val_acc": val_acc,
    #                     "val_ece": val_ece,
    #                     "val_nll": val_nll,
    #                     "std": std,
    #                     "val_brier": val_brier,
    #                 }
    #             )
    #     return val_acc, val_ece, val_nll, val_brier

    def evaluate(self, test_loader, val_loader):
        print("EVALUATing started!")
        print("self.eval_n_samples:", self.eval_n_samples)
        self.eval()
        status = self.training

        import numpy as np
        import matplotlib.pyplot as plt
        import os

        def compute_ece(probs, labels, split, n_bins=10, plot=True):
            save_path = f"/kaggle/working/Plots/{self.args.modelwrapper}_{self.args.model.split('/')[1]}_{self.args.dataset}_{self.args.max_train_steps}_{split}.png"
            # Ensure inputs are NumPy arrays
            probs = np.array(probs)
            labels = np.array(labels)

            # Compute confidences and predictions
            confidences = probs.max(axis=1)
            predictions = probs.argmax(axis=1)
            accuracies = (predictions == labels)

            # Initialize ECE
            ece = 0.0
            bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

            # For plotting
            bin_centers = []
            avg_confs = []
            avg_accs = []
            abs_diffs = []

            for i in range(n_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                prop_in_bin = np.mean(in_bin)

                if prop_in_bin > 0:
                    acc_in_bin = np.mean(accuracies[in_bin])
                    avg_conf = np.mean(confidences[in_bin])
                    ece += np.abs(avg_conf - acc_in_bin) * prop_in_bin

                    # Store for plotting
                    bin_center = (bin_lower + bin_upper) / 2
                    bin_centers.append(bin_center)
                    avg_confs.append(avg_conf)
                    avg_accs.append(acc_in_bin)
                    abs_diffs.append(np.abs(avg_conf - acc_in_bin))

            if plot:
                plt.figure(figsize=(8, 6))
                plt.plot(bin_centers, avg_confs, label='Confidence', marker='o')
                plt.plot(bin_centers, avg_accs, label='Accuracy', marker='x')
                plt.bar(bin_centers, abs_diffs, width=1/n_bins, alpha=0.3, label='|Conf - Acc|')
                plt.xlabel('Confidence Bin Center')
                plt.ylabel('Value')
                plt.title(f'Calibration Plot for {split} Set (ECE = {ece:.4f})')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
        
                # Save the figure if save_path is provided
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path)
                    print(f"Plot saved to: {save_path}")
            
                plt.show()

            return ece
        
        def _run_eval(eval_loader, name="eval"):
            nlls = AverageMeter()
            briers = AverageMeter()
            metric_kwargs = {"task": "multiclass", "num_classes": self.num_classes}
            acc_metric = Accuracy(**metric_kwargs).to(self.accelerator.device)
            ece_metric = CalibrationError(**metric_kwargs, n_bins=self.args.num_bins).to(self.accelerator.device)

            all_probs = []
            all_preds = []
            all_labels = []

            samples_seen = 0
            progress_bar = tqdm(eval_loader, desc=f"Evaluating {name}", leave=True)
            for step, batch in enumerate(progress_bar):
                with torch.no_grad() and torch.inference_mode():
                    logits = self.forward_logits(
                        batch,
                        sample=not self.args.bayes_inference_notsample,
                        n_samples=self.eval_n_samples,
                    ).detach()

                    if self.args.dataset_type == "mcdataset":
                        _, labels, _ = batch
                    else:
                        labels = batch["labels"]

                    logits, labels = self.accelerator.gather([logits, labels])
                    if self.accelerator.num_processes > 1:
                        if step == len(eval_loader) - 1:
                            labels = labels[: len(eval_loader.dataset) - samples_seen]
                            logits = logits[: len(eval_loader.dataset) - samples_seen]
                        else:
                            samples_seen += labels.shape[0]

                    probs = torch.softmax(logits, dim=-1).mean(dim=1)
                    std = torch.softmax(logits, dim=-1).std(dim=1).mean()

                    acc_metric(probs, labels)
                    ece_metric(probs, labels)
                    nll = self.loss(torch.log(probs), labels, reduction="mean")

                    if torch.isnan(nll):
                        if self.accelerator.is_local_main_process:
                            print("NaN NLL detected")
                            print("nll:", nll)
                            print("probs:", probs)
                            print("logits:", logits)
                            exit()

                    nlls.update(nll)

                    brier = (
                        (probs - F.one_hot(labels, num_classes=logits.size(-1)))
                        .pow(2)
                        .sum(dim=-1)
                        .mean()
                    )
                    briers.update(brier)

                    pred_id = probs.argmax(dim=-1)
                    true_id = labels

                    all_probs.append(probs.cpu().numpy())
                    all_preds.append(pred_id.cpu().numpy())
                    all_labels.append(true_id.cpu().numpy())

                    progress_bar.set_postfix({
                        "acc": acc_metric.compute().item(),
                        "nll": nlls.avg,
                        "brier": briers.avg,
                    })

                # if step == 10:
                #     break
            
            acc = acc_metric.compute().item()
            ece = ece_metric.compute().item()
            nll = nlls.avg
            brier = briers.avg

            if self.accelerator.is_local_main_process and self.wandb_logger is not None:
                self.wandb_logger.log({
                    f"{name}_acc": acc,
                    f"{name}_ece": ece,
                    f"{name}_nll": nll,
                    f"{name}_brier": brier,
                    f"{name}_std": std,
                })

            all_probs = np.concatenate(all_probs, axis=0)
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            return (acc, ece, nll, brier), (all_probs, all_preds, all_labels)
            
        val_metrics = None
        if val_loader:
            val_metrics, all_val_probs = _run_eval(val_loader, name="val")
            all_probs, all_preds, all_labels = all_val_probs
            _ = compute_ece(all_probs, all_labels, "Validation(OOD)", n_bins=10, plot=True)
            
        test_metrics, all_test_probs = _run_eval(test_loader, name="test")
        all_probs, all_preds, all_labels = all_test_probs
        _ = compute_ece(all_probs, all_labels, "Test(ID)", n_bins=10, plot=True)
        self.train(status)
        print(val_metrics, test_metrics)
        return val_metrics, test_metrics

    def fit_evaluate(self):
        """Performs the fitting and evaluation process, saving the results to checkpoints 
        and logging them to the WandB logger.
        """
        print("FITTEVALUATing!")
        if self.accelerator.is_local_main_process:
            save_folder = f"checkpoints/{self.args.modelwrapper}/{self.args.model}/{self.args.dataset}/{self.args.log_path}"
            create_if_not_exists(save_folder)
            logging.basicConfig(
                format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
                level=logging.INFO,
                filename=save_folder + "/log.txt",
            )

        for epoch in range(self.args.n_epochs):
            print(f"Epoch {epoch}!")
            if self.args.early_stop_steps > 0 and epoch >= self.earlystop_n_epochs:
                break
            self.args.epoch = epoch
            self.fit(self.train_loader, self.test_loader, self.val_loader)
            val_metrics, test_metrics = self.evaluate(self.test_loader, self.val_loader)

        if hasattr(self.args, "bayes_eval_n_samples_final"):
            self.eval_n_samples = self.args.bayes_eval_n_samples_final

        # val_metrics, test_metrics = self.evaluate(self.test_loader, self.val_loader)
        test_acc, test_ece, test_nll, test_brier = test_metrics
        if val_metrics:
            val_acc, val_ece, val_nll, val_brier = val_metrics
            logging.info(
            f"val_acc: {val_acc}, val_ece: {val_ece}, val_nll: {val_nll}, val_brier: {val_brier}, test_acc: {test_acc}, test_ece: {test_ece}, test_nll: {test_nll}, test_brier: {test_brier}")
        else:
            logging.info(
            f"test_acc: {test_acc}, test_ece: {test_ece}, test_nll: {test_nll}, test_brier: {test_brier}")
        if self.accelerator.is_local_main_process:
            if self.wandb_logger is not None:
                if val_metrics:
                    self.wandb_logger.log(
                        {
                            "final_test_acc": test_acc,
                            "final_test_ece": test_ece,
                            "final_test_nll": test_nll,
                            "final_test_brier": test_brier,
                            "final_val_acc": val_acc,
                            "final_val_ece": val_ece,
                            "final_val_nll": val_nll,
                            "final_val_brier": val_brier,
                        }
                    )
                else:
                    self.wandb_logger.log(
                        {
                            "final_test_acc": test_acc,
                            "final_test_ece": test_ece,
                            "final_test_nll": test_nll,
                            "final_test_brier": test_brier,
                        }
                    )

    def prepare_for_fit_evaluate(self, dataset, wandb_logger=None):
        """
        Prepare the model for training and evaluation.
        """
        print("PREPARing started!")
        self.wandb_logger = wandb_logger
        train_loader, test_loader = dataset.train_dataloader, dataset.test_dataloader
        if self.args.testing_set == "train_val":
            val_loader = dataset.val_dataloader
            val_loader = self.accelerator.prepare(val_loader)
            self.val_loader = val_loader

        if isinstance(dataset, BOSSDataset):
            print("The dataset is an instance of BOSSDataset")
            val_loader = dataset.val_dataloader
            val_loader = self.accelerator.prepare(val_loader)
            self.val_loader = val_loader
        
        if self.args.dataset_type == "mcdataset":
            self.target_ids = dataset.target_ids.squeeze(-1)

        l_train = len(train_loader)

        num_update_steps_per_epoch = len(train_loader)
        if self.args.max_train_steps == 0:
            self.args.max_train_steps = self.args.n_epochs * num_update_steps_per_epoch
        self.args.n_epochs = (
            math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)
            if self.args.ood_ori_dataset is None
            else 0
        )
        if self.args.early_stop_steps > 0:
            self.earlystop_n_epochs = (
                math.ceil(self.args.early_stop_steps / num_update_steps_per_epoch)
                if self.args.ood_ori_dataset is None
                else 0
            )
        else:
            self.earlystop_n_epochs = 0
        if self.accelerator.is_local_main_process:
            print("len(train_loader):", len(train_loader))
            print("num of epochs:", self.args.n_epochs)
        self.step = 0

        (
            self.base_model,
            self.opt,
            train_loader,
            test_loader,
            self.scheduler,
            self.scheduler2,
            self.opt2,
        ) = self.accelerator.prepare(
            self.base_model,
            self.opt,
            train_loader,
            test_loader,
            self.scheduler,
            self.scheduler2,
            self.opt2,
        )

        self.train_loader = train_loader
        self.test_loader = test_loader
        if self.args.bayes_datasetrescaling:
            self.M = int(
                100
                * (dataset.num_samples ** (math.pi / self.args.bayes_gamma))
                / (l_train / len(train_loader))
                / self.args.batch_size
            )
        else:
            self.M = len(train_loader)

        print("M:", self.M)
