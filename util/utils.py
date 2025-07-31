import argparse
from contextlib import redirect_stdout
import itertools
import logging
import math
import os
from datetime import datetime
from typing import List, Union

import numpy as np
import PIL.Image
import safetensors.torch
import torch
import torch.nn.functional as F
import tqdm
import imageio
import subprocess
from PIL import Image
from spandrel import ModelLoader

from diffusers.utils import export_to_video


logger = logging.getLogger(__file__)
def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for ConsisID.")

    # ConsisID information
    parser.add_argument("--train_type", choices=['t2v', 'i2v'], help="Training type: t2v or i2v")
    parser.add_argument("--is_train_face", action='store_true', help="Whether to train on face data")
    parser.add_argument("--is_diff_lr", action='store_true', help="Whether to use different learning rates")
    parser.add_argument("--is_train_lora", action='store_true', help="Whether to train with LoRA")
    parser.add_argument("--is_kps", action='store_true', help="Whether to use keypoints")
    parser.add_argument("--is_shuffle_data", action='store_true', help="Whether to shuffle the data")
    parser.add_argument("--enable_mask_loss", action='store_true', help="Whether to enable mask loss")
    parser.add_argument("--is_single_face", action='store_true', help="Whether to use a single face")
    parser.add_argument("--is_cross_face", action='store_true', help="Whether to use cross-face data")
    parser.add_argument("--is_align_face", action='store_true', help="Whether to align faces")
    parser.add_argument("--is_reserve_face", action='store_true', help="Whether to reserve face data")
    parser.add_argument("--is_accelerator_state_dict", action='store_true', help="Whether to use accelerator state dictionary")
    parser.add_argument("--is_validation", action='store_true', help="Whether to perform validation")
    parser.add_argument("--config_path", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--mask_path", type=str, default=None, help="Path to mask file")
    parser.add_argument("--pretrained_weight", type=str, default=None, help="Path to pretrained weights")
    parser.add_argument("--sample_stride", type=int, default=3, help="Sample stride")
    parser.add_argument("--skip_frames_start_percent", type=float, default=0.0, help="Percentage of frames to skip at the start")
    parser.add_argument("--skip_frames_end_percent", type=float, default=1.0, help="Percentage of frames to skip at the end")
    parser.add_argument("--miss_tolerance", type=int, default=6, help="Tolerance for missing frames")
    parser.add_argument("--min_distance", type=int, default=3, help="Minimum distance between faces")
    parser.add_argument("--min_frames", type=int, default=1, help="Minimum number of frames")
    parser.add_argument("--max_frames", type=int, default=5, help="Maximum number of frames")
    parser.add_argument("--cross_attn_interval", type=int, default=2, help="Interval between cross-attention layers")
    parser.add_argument("--cross_attn_dim_head", type=int, default=128, help="Dimensionality of each attention head in cross-attention")
    parser.add_argument("--cross_attn_num_heads", type=int, default=16, help="Number of attention heads in cross-attention")
    parser.add_argument("--LFE_id_dim", type=int, default=1280, help="Dimensionality of the identity vector in LFE")
    parser.add_argument("--LFE_vit_dim", type=int, default=1024, help="Dimensionality of the Vision Transformer output in LFE")
    parser.add_argument("--LFE_depth", type=int, default=10, help="Number of layers in LFE")
    parser.add_argument("--LFE_dim_head", type=int, default=64, help="Dimensionality of each attention head in LFE")
    parser.add_argument("--LFE_num_heads", type=int, default=16, help="Number of attention heads in LFE")
    parser.add_argument("--LFE_num_id_token", type=int, default=5, help="Number of identity tokens in LFE")
    parser.add_argument("--LFE_num_querie", type=int, default=32, help="Number of query tokens in LFE")
    parser.add_argument("--LFE_output_dim", type=int, default=2048, help="Output dimension of LFE")
    parser.add_argument("--LFE_ff_mult", type=int, default=4, help="Multiplication factor for feed-forward network in LFE")
    parser.add_argument("--LFE_num_scale", type=int, default=5, help="The number of different scales visual feature in LFE")
    parser.add_argument("--local_face_scale", type=float, default=1.0, help="Scaling factor for local facial features")
    parser.add_argument("--mask_prob", type=float, default=0.5, help="Scaling factor for local facial features")
    parser.add_argument("--drop_inpaint_prob", type=float, default=0.5, help="Drop probability for inpaint")
    parser.add_argument("--routing_logits_zeros_prob", type=float, default=0, help="Drop probability for routing logits")
    parser.add_argument("--is_train_audio", action='store_true', help="Whether to train with audio")
    parser.add_argument("--router_loss_weight", type=float, default=1, help="Weight for router loss")
    parser.add_argument("--consistency_loss_weight", type=float, default=0.1, help="Weight for consistency loss")
    parser.add_argument("--temporal_diff_loss_weight", type=float, default=0.1, help="Weight for temporal difference loss")
    parser.add_argument("--spatial_diff_loss_weight", type=float, default=0.01, help="Weight for spatial difference loss")
    parser.add_argument("--spatial_dist_loss_weight", type=float, default=1, help="Weight for spatial distribution loss")
    parser.add_argument("--id_dist_loss_weight", type=float, default=1, help="Weight for ID distribution loss")
    parser.add_argument("--load_pretrained_module", action='store_true', help="Whether to load pretrained module(audio+face+lora)")
    parser.add_argument("--load_pretrained_modules_list", type=str, nargs='+', default=None, help="Whether to load pretrained module(audio+face+lora)")
    parser.add_argument("--load_pretrained_modules_list_path", type=str, nargs='+', default=None, help="Whether to load pretrained module(audio+face+lora)")
    parser.add_argument("--is_teacher_forcing", action='store_true', help="Whether to use teacher forcing")
    parser.add_argument("--index_mask_drop_prob", type=float, default=0.0, help="Drop probability for index mask")
    # EMA
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )

    # Model information
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    # Dataset information
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_root",
        type=str,
        default=None,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--video_column",
        type=str,
        default="video",
        help="The column of the dataset containing videos. Or, the name of the file in `--instance_data_root` folder containing the line-separated path to video data.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing the instance prompt for each video. Or, the name of the file in `--instance_data_root` folder containing the line-separated instance prompts.",
    )
    parser.add_argument(
        "--id_token", type=str, default=None, help="Identifier token appended to the start of each prompt if provided."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    # Validation
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="One or more prompt(s) that is used during validation to verify that the model is learning. Multiple validation prompts should be separated by the '--validation_prompt_seperator' string.",
    )
    parser.add_argument(
        "--validation_images",
        type=str,
        default=None,
        help="One or more image path(s) that is used during validation to verify that the model is learning. Multiple validation paths should be separated by the '--validation_prompt_seperator' string. These should correspond to the order of the validation prompts.",
    )
    parser.add_argument(
        "--validation_prompt_separator",
        type=str,
        default=":::",
        help="String that separates multiple validation prompts",
    )
    parser.add_argument(
        "--num_validation_videos",
        type=int,
        default=1,
        help="Number of videos that should be generated during validation per `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run validation every X epochs. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_videos`."
        ),
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6,
        help="The guidance scale to use while sampling validation videos.",
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        default=False,
        help="Whether or not to use the default cosine dynamic guidance schedule when sampling validation videos.",
    )

    # Training information
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=128,
        help=("The scaling factor to scale LoRA weight update. The actual scaling factor is `lora_alpha / rank`"),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cogvideox-i2v-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="All input videos are resized to this height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="All input videos are resized to this width.",
    )
    parser.add_argument("--in_channels", type=int, default=48, help="Number of input channels")
    parser.add_argument("--fps", type=int, default=8, help="All input videos will be used at this FPS.")
    parser.add_argument(
        "--max_num_frames", type=int, default=49, help="All input videos will be truncated to these many frames."
    )
    parser.add_argument(
        "--skip_frames_start",
        type=int,
        default=0,
        help="Number of frames to skip from the beginning of each input video. Useful if training data contains intro sequences.",
    )
    parser.add_argument(
        "--skip_frames_end",
        type=int,
        default=0,
        help="Number of frames to skip from the end of each input video. Useful if training data contains outro sequences.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip videos horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine_with_restarts",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",
    )
    parser.add_argument(
        "--noised_image_dropout",
        type=float,
        default=0.05,
        help="Image condition dropout probability.",
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw", "prodigy"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    parser.add_argument("--prodigy_decouple", action="store_true", help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Turn on Adam's bias correction.")
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
    )
    # unfreeze_modules
    parser.add_argument(
        '--unfreeze_modules',
        nargs='+',
        help='Enter a list of unfreeze modules'
    )
    # freeze_modules
    parser.add_argument(
        '--freeze_modules',
        nargs='+',
        help='Enter a list of freeze modules'
    )
    

    # Other information
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        '--trainable_modules',
        nargs='+',
        help='Enter a list of trainable modules'
    )
    parser.add_argument("--nccl_timeout", type=int, default=600, help="NCCL backend timeout in seconds.")
    parser.add_argument(
        "--step_timeout",
        type=int,
        default=180,
        help="Timeout in seconds for each training step. Default is 180 seconds (3 minutes).",
    )

    return parser.parse_args()

def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )

        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask

def save_tensor_as_image(tensor, file_path):
    """
    Saves a PyTorch tensor as an image file.

    Args:
        tensor (torch.Tensor): The image tensor to save.
        file_path (str): Path to save the image file.
    """
    # Ensure the tensor is in CPU memory and detach it from the computation graph
    tensor = tensor.cpu().detach()

    # Convert from PyTorch to NumPy format, and handle the scaling from [0, 1] to [0, 255]
    tensor = tensor.squeeze()  # Remove unnecessary dimensions if any
    tensor = tensor.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
    tensor = tensor.numpy() * 255  # Scale from [0, 1] to [0, 255]
    tensor = tensor.astype(np.uint8)  # Convert to uint8

    # Convert the NumPy array to a PIL Image and save it
    image = Image.fromarray(tensor)
    image.save(file_path)

def pixel_values_to_pil(pixel_values, frame_index=0):
    if pixel_values.is_cuda:
        pixel_values = pixel_values.clone().cpu()
    pixel_values = (pixel_values + 1.0) / 2.0 * 255.0
    pixel_values = pixel_values.clamp(0, 255).byte()
    frame = pixel_values[frame_index]  # [C, H, W]
    frame = frame.permute(1, 2, 0)  # [H, W, C]
    frame_np = frame.numpy()
    image = Image.fromarray(frame_np)
    return image

def load_torch_file(ckpt, device=None, dtype=torch.float16):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        if "weights_only" not in torch.load.__code__.co_varnames:
            logger.warning(
                "Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely."
            )

        pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        if "global_step" in pl_sd:
            logger.debug(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        elif "params_ema" in pl_sd:
            sd = pl_sd["params_ema"]
        else:
            sd = pl_sd

    sd = {k: v.to(dtype) for k, v in sd.items()}
    return sd


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = [
            (a, "{}{}".format(replace_prefix[rp], a[len(rp):]))
            for a in filter(lambda a: a.startswith(rp), state_dict.keys())
        ]
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


def module_size(module):
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    return module_mem


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))


@torch.inference_mode()
def tiled_scale_multidim(
    samples, function, tile=(64, 64), overlap=8, upscale_amount=4, out_channels=3, output_device="cpu", pbar=None
):
    dims = len(tile)
    print(f"samples dtype:{samples.dtype}")
    output = torch.empty(
        [samples.shape[0], out_channels] + [round(a * upscale_amount) for a in samples.shape[2:]],
        device=output_device,
    )

    for b in range(samples.shape[0]):
        s = samples[b : b + 1]
        out = torch.zeros(
            [s.shape[0], out_channels] + [round(a * upscale_amount) for a in s.shape[2:]],
            device=output_device,
        )
        out_div = torch.zeros(
            [s.shape[0], out_channels] + [round(a * upscale_amount) for a in s.shape[2:]],
            device=output_device,
        )

        for it in itertools.product(*(range(0, a[0], a[1] - overlap) for a in zip(s.shape[2:], tile))):
            s_in = s
            upscaled = []

            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap, it[d]))
                l = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(pos * upscale_amount))

            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)
            feather = round(overlap * upscale_amount)
            for t in range(feather):
                for d in range(2, dims + 2):
                    m = mask.narrow(d, t, 1)
                    m *= (1.0 / feather) * (t + 1)
                    m = mask.narrow(d, mask.shape[d] - 1 - t, 1)
                    m *= (1.0 / feather) * (t + 1)

            o = out
            o_d = out_div
            for d in range(dims):
                o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])

            o += ps * mask
            o_d += mask

            if pbar is not None:
                pbar.update(1)

        output[b : b + 1] = out / out_div
    return output


def tiled_scale(
    samples,
    function,
    tile_x=64,
    tile_y=64,
    overlap=8,
    upscale_amount=4,
    out_channels=3,
    output_device="cpu",
    pbar=None,
):
    return tiled_scale_multidim(
        samples, function, (tile_y, tile_x), overlap, upscale_amount, out_channels, output_device, pbar
    )


def load_sd_upscale(ckpt, inf_device):
    sd = load_torch_file(ckpt, device=inf_device)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = state_dict_prefix_replace(sd, {"module.": ""})
    out = ModelLoader().load_from_state_dict(sd).half()
    return out


def upscale(upscale_model, tensor: torch.Tensor, inf_device, output_device="cpu") -> torch.Tensor:
    memory_required = module_size(upscale_model.model)
    memory_required += (
        (512 * 512 * 3) * tensor.element_size() * max(upscale_model.scale, 1.0) * 384.0
    )  # The 384.0 is an estimate of how much some of these models take, TODO: make it more accurate
    memory_required += tensor.nelement() * tensor.element_size()
    print(f"UPScaleMemory required: {memory_required / 1024 / 1024 / 1024} GB")

    upscale_model.to(inf_device)
    tile = 512
    overlap = 32

    steps = tensor.shape[0] * get_tiled_scale_steps(
        tensor.shape[3], tensor.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
    )

    pbar = ProgressBar(steps, desc="Tiling and Upscaling")

    s = tiled_scale(
        samples=tensor.to(torch.float16),
        function=lambda a: upscale_model(a),
        tile_x=tile,
        tile_y=tile,
        overlap=overlap,
        upscale_amount=upscale_model.scale,
        pbar=pbar,
    )

    upscale_model.to(output_device)
    return s


def upscale_batch_and_concatenate(upscale_model, latents, inf_device, output_device="cpu") -> torch.Tensor:
    upscaled_latents = []
    for i in range(latents.size(0)):
        latent = latents[i]
        upscaled_latent = upscale(upscale_model, latent, inf_device, output_device)
        upscaled_latents.append(upscaled_latent)
    return torch.stack(upscaled_latents)


def save_video(tensor: Union[List[np.ndarray], List[PIL.Image.Image]], fps: int = 8):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"./assets/output/{timestamp}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    export_to_video(tensor, video_path, fps=fps)
    return video_path


class ProgressBar:
    def __init__(self, total, desc=None):
        self.total = total
        self.current = 0
        self.b_unit = tqdm.tqdm(total=total, desc="ProgressBar context index: 0" if desc is None else desc)

    def update(self, value):
        if value > self.total:
            value = self.total
        self.current = value
        if self.b_unit is not None:
            self.b_unit.set_description("ProgressBar context index: {}".format(self.current))
            self.b_unit.refresh()

            self.b_unit.update(self.current)


def save_frames_to_png_mp4(tensor, save_dir,video_save_dir=None):
    img_list=[]
    os.makedirs(save_dir, exist_ok=True)
    for i in range(tensor.size(0)):
        frame = tensor[i]
        normalized = frame

        uint8_frame = (normalized * 255).type(torch.uint8)
        img = Image.fromarray(uint8_frame.numpy(), mode='L')
        # resize from 45*30 to 720*480
        img = img.resize((720, 480), Image.Resampling.NEAREST)
        img_path = f"{save_dir}/frame_{i:02d}.png"
        img.save(img_path)
        for i in range(4 if i!=0 else 1):
            img_list.append(img)
    # save these pngs merged into mp4
    if video_save_dir is None:
        video_save_dir=save_dir+"/output.mp4"
    imageio.mimsave(video_save_dir, img_list, fps=25)


def draw_routing_logit(routing_logits, base_dir="assets/output/tempfile", suffix="",video_save_dir=None,use_softmax=True):
    routing_logit=routing_logits[1] # bs,bs1 is uncond, not meaningful, take the second batch i.e. full condition
    torch.save(routing_logit, base_dir+"routing_logit"+suffix+".pt")
    if use_softmax:
        routing_logit = torch.softmax(routing_logit.float(), dim=-1).to(routing_logit.dtype)
    routing_logit = routing_logit.squeeze(0).cpu().float()
    logit_0 = routing_logit[:, 0].view(13, 30, 45)
    logit_1 = routing_logit[:, 1].view(13, 30, 45)
    prefix="routing_logit_0"
    video_save_dir_0 = video_save_dir.replace(".mp4", "_0.mp4")
    save_frames_to_png_mp4(logit_0, base_dir+prefix+suffix,video_save_dir_0)
    prefix="routing_logit_1"
    video_save_dir_1 = video_save_dir.replace(".mp4", "_1.mp4")
    save_frames_to_png_mp4(logit_1, base_dir+prefix+suffix,video_save_dir_1)


def merge_audio_video(audio_path, video_path, output_path, time_to_skip_audio=0, time_to_skip_video=0, skip_first_frame=False):
    """
    Use ffmpeg to merge audio and video, supporting separate trimming of audio and video.

    :param time_to_skip_audio: Start time (in seconds) to skip in the audio
    :param time_to_skip_video: Start time (in seconds) to skip in the video
    :param skip_first_frame: Whether to enable audio and video trimming
    """
    temp_video = None
    temp_audio = None

    try:
        if skip_first_frame:
            temp_video = video_path + "_temp.mp4"
            video_trim_command = [
                "ffmpeg",
                "-y",
                "-ss", str(time_to_skip_video),
                "-i", video_path,
                "-c:v", "copy",
                temp_video
            ]
            subprocess.run(video_trim_command, check=True)
            video_input = temp_video

            temp_audio = audio_path + "_temp.wav"
            audio_trim_command = [
                "ffmpeg",
                "-y",
                "-i", audio_path,
                "-ss", str(time_to_skip_audio),
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                temp_audio
            ]
            subprocess.run(audio_trim_command, check=True)
            audio_input = temp_audio
        else:
            video_input = video_path
            audio_input = audio_path

        merge_command = [
            "ffmpeg",
            "-y",
            "-i", video_input,
            "-i", audio_input,
            "-c:v", "copy",
            "-c:a", "aac",
            "-ar", "16000",
            "-shortest",
            output_path
        ]
        subprocess.run(merge_command, check=True)

        print(f"合成完成，输出文件为：{output_path}")

    except subprocess.CalledProcessError as e:
        print("合成过程中出错：", e)
    finally:
        if skip_first_frame:
            if temp_video and os.path.exists(temp_video):
                os.remove(temp_video)
            if temp_audio and os.path.exists(temp_audio):
                os.remove(temp_audio)


def process_single_mask_dir(mask_dir):
    """Process masks from a single directory"""
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
    dense_masks_dict = []

    for frame in range(len(mask_files)):
        mask_path = os.path.join(mask_dir, f"annotated_frame_{int(frame):05d}.png")
        mask_array = np.array(Image.open(mask_path))
        binary_mask = np.where(mask_array > 0, 1, 0).astype(np.uint8)
        dense_masks_dict.append(binary_mask)

    # Convert to tensor
    dense_masks = torch.stack([torch.from_numpy(m) for m in dense_masks_dict])  # [T, H, W]
    dense_masks = dense_masks.unsqueeze(0)  # [B=1, T, H, W]

    return dense_masks


def process_masks_to_routing_logits(base_dir):
    """Process masks to teacher routing logits following the original codebase exactly"""

    # ----------------- Part 1: Check directories and load masks -----------------
    dir1 = os.path.join(base_dir, "1")
    dir2 = os.path.join(base_dir, "2")

    if not (os.path.exists(dir1) and os.path.exists(dir2)):
        raise ValueError(f"both subdirectories '1' and '2' must exist in {base_dir}")

    # Load masks from both directories
    dense_masks_1 = process_single_mask_dir(dir1)
    dense_masks_2 = process_single_mask_dir(dir2)

    # ----------------- Part 2: Process masks (from train.py) -----------------
    B = 1  # Fixed batch size
    T = 13  # Fixed temporal length
    H = 60  # Fixed height
    W = 90  # Fixed width
    p = 2   # Fixed patch size

    # Process masks from directory 1
    current_mask_1 = dense_masks_1.to(memory_format=torch.contiguous_format).float()
    current_mask_1 = current_mask_1.unsqueeze(1)  # [B, 1, T, H, W]

    # Process masks from directory 2
    current_mask_2 = dense_masks_2.to(memory_format=torch.contiguous_format).float()
    current_mask_2 = current_mask_2.unsqueeze(1)  # [B, 1, T, H, W]

    # Create fake latent tensor
    fake_latent = torch.zeros((B, 1, T, H//p, W//p))

    # Initialize index_mask as background (-1)
    index_mask = torch.full((B, 1, T, H//p, W//p), -1, dtype=torch.long)

    # Resize and process mask 1
    resized_mask_1 = resize_mask(current_mask_1, fake_latent, process_first_frame_only=False)
    binary_mask_1 = (resized_mask_1 > 0.5).long()

    # Resize and process mask 2
    resized_mask_2 = resize_mask(current_mask_2, fake_latent, process_first_frame_only=False)
    binary_mask_2 = (resized_mask_2 > 0.5).long()

    # Fill index_mask with labels:
    # - background (-1) -> [0, 1]
    # - mask 1 (1) -> [0, 0]
    # - mask 2 (1) -> [1, 0]
    index_mask = torch.where(binary_mask_1 == 1, torch.tensor(0, dtype=torch.long), index_mask)
    index_mask = torch.where(binary_mask_2 == 1, torch.tensor(1, dtype=torch.long), index_mask)

    # Remove channel dimension and reshape
    index_mask = index_mask.squeeze(1)  # [B, T, H//p, W//p]
    index_mask = index_mask.reshape(B, -1)  # [B, len]

    # ----------------- Part 3: Generate teacher routing logits -----------------
    routing_logit_shape = (1, index_mask.shape[1], 2)  # [1, T*H//p*W//p, 2]
    teacher_routing_logit = torch.zeros(routing_logit_shape)

    # Set logits according to mask values:
    # background (-1) -> [0, 0]
    # mask 1 (0) -> [1, 0]
    # mask 2 (1) -> [0, 1]
    teacher_routing_logit[0, index_mask[0] == 0, 0] = 1   # mask 1
    teacher_routing_logit[0, index_mask[0] == 1, 1] = 1   # mask 2

    return teacher_routing_logit


def get_routing_logits_from_tracking_mask_results(tracking_mask_results_dir,is_draw_video=True,video_save_dir="assets/output/tempfile/temp_mask.mp4"):
    routing_logits = process_masks_to_routing_logits(tracking_mask_results_dir)

    if is_draw_video:
        from util.utils import draw_routing_logit
        routing_logits_draw=[ None ,routing_logits]
        video_dir = os.path.dirname(video_save_dir)
        if video_dir and not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)
        draw_routing_logit(routing_logits=routing_logits_draw,suffix="test1",video_save_dir=video_save_dir,use_softmax=False)

    return routing_logits


def load_router_from_transformer_safetensors(transformer, transformer_path,log_file_path=None):
    from safetensors.torch import load_file

    # 0. load weights from safetensors
    folder = os.path.join(transformer_path, "transformer")
    index_file = os.path.join(folder, "diffusion_pytorch_model.safetensors.index.json")
    with open(index_file, "r") as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    all_shards = set(weight_map.values())
    tensors_initial = {}
    for shard in all_shards:
        shard_path = os.path.join(folder, shard)
        shard_tensors = load_file(shard_path)
        tensors_initial.update(shard_tensors)

    # 1. load router weights to transformer (only router weights)
    tensors_router = {}
    for key in tensors_initial.keys():
        if "router" in key:
            # tensors_lora[key] = tensors_initial[key]
            # transformer_blocks.改为base_model.model.transformer_blocks.
            key_new = key.replace("router.", "base_model.model.router.")
            tensors_router[key_new] = tensors_initial[key]

    with open(log_file_path, 'a') as f:
        with redirect_stdout(f):
            print(f"--------------------------------start loading router from transformer_safetensors--------------------------------")
            missing_keys, unexpected_keys = transformer.load_state_dict(tensors_router, strict=False)
            print(f'unexpected_keys: {unexpected_keys}')
            # print(f'missing_keys: {missing_keys}')
            print(f"--------------------------------complete loading router from transformer_safetensors--------------------------------")
    return transformer


def load_lora_from_transformer_safetensors(transformer, transformer_path, lora_rank=128,flag_lora_ready=False,log_file_path=None):
    from safetensors.torch import load_file
    from peft import LoraConfig, get_peft_model
    # 0. load weights from safetensors
    folder = os.path.join(transformer_path, "transformer")
    index_file = os.path.join(folder, "diffusion_pytorch_model.safetensors.index.json")
    with open(index_file, "r") as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    all_shards = set(weight_map.values())
    tensors_initial = {}
    for shard in all_shards:
        shard_path = os.path.join(folder, shard)
        shard_tensors = load_file(shard_path)
        tensors_initial.update(shard_tensors)

    # 1. init lora for transformer
    if not flag_lora_ready:
        transformer_lora_config = LoraConfig(r=lora_rank,lora_alpha=128,init_lora_weights=True,target_modules=["attn1.to_k", "attn1.to_q"])
        transformer = get_peft_model(transformer, transformer_lora_config)
    # 2. load lora weights to transformer (only lora weights)
    tensors_lora = {}
    for key in tensors_initial.keys():
        if "lora" in key:
            # tensors_lora[key] = tensors_initial[key]
            # transformer_blocks.改为base_model.model.transformer_blocks.
            key_new = key.replace("transformer_blocks.", "base_model.model.transformer_blocks.")
            tensors_lora[key_new] = tensors_initial[key]

    with open(log_file_path, 'a') as f:
        with redirect_stdout(f):
            print(f"--------------------------------start loading lora from transformer_safetensors--------------------------------")
            missing_keys, unexpected_keys = transformer.load_state_dict(tensors_lora, strict=False)
            print(f'unexpected_keys: {unexpected_keys}')
            print(f'missing_keys length: {len(missing_keys)}')
            print(f"--------------------------------complete loading lora from transformer_safetensors--------------------------------")
    return transformer


def load_mixed_lora_weights(transformer, lora_paths, lora_rank=128,log_file_path=None):
    from safetensors.torch import load_file
    from peft import get_peft_model
    from peft import LoraConfig, get_peft_model
    for lora_path in lora_paths:
        lora_weights = load_file(lora_path)
        lora_weights = {k.replace("transformer.module.", "transformer."): v  for k, v in lora_weights.items()}
        lora_weights = {k.replace("transformer.", "base_model.model."): v  for k, v in lora_weights.items()}
        lora_weights = {k.replace("lora_A", "lora_A.default"): v  for k, v in lora_weights.items()}
        lora_weights = {k.replace("lora_B", "lora_B.default"): v  for k, v in lora_weights.items()}

        transformer_lora_config = LoraConfig(r=lora_rank,lora_alpha=128,init_lora_weights=True,target_modules=["attn1.to_k", "attn1.to_q"])

        transformer = get_peft_model(transformer, transformer_lora_config)
        missing_keys, unexpected_keys = transformer.load_state_dict(    lora_weights,     strict=False)
        with open(log_file_path, 'a') as f:
            with redirect_stdout(f):
                print(f"--------------------------------start loading lora from safetensors--------------------------------")
                print(f'unexpected_keys: {unexpected_keys}')
                print(f'missing_keys len: {len(missing_keys)}')
                print(f"--------------------------------complete loading lora from safetensors--------------------------------")
    return transformer
