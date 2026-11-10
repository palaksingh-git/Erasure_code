"""Partially denoised student latents and score distillation objectives."""
import random

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from peft import LoraConfig

from .model import encode


TEMPLATES = (
    "a photo of {}", "a candid shot of {}", "a painting of {}", "a close-up photo of {}",
    "an illustration of {}", "an oil painting of {}", "an anime-style drawing of {}",
    "a detailed portrait of {}", "a studio photograph of {}", "a watercolor of {}",
)


def teacher_unet(cfg, device, checkpoint=None):
    kwargs = {"subfolder": "unet"} if checkpoint is None else {}
    if checkpoint is None and cfg.get("variant"):
        kwargs["variant"] = cfg["variant"]
    source = cfg["model_id"] if checkpoint is None else str(checkpoint)
    teacher = UNet2DConditionModel.from_pretrained(source, torch_dtype=torch.float32, **kwargs).to(device)
    teacher.eval().requires_grad_(False)
    return teacher


@torch.no_grad()
def partial_student_latent(pipe, conditioning, denoise_steps):
    """Draw x_t by running the current student from Gaussian noise to a random DDIM step."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(denoise_steps, device=pipe.device)
    timesteps = scheduler.timesteps
    if len(timesteps) < 2:
        raise ValueError("denoise_steps must be at least 2")
    stop = random.randrange(1, len(timesteps))
    shape = (1, pipe.unet.config.in_channels, pipe.unet.config.sample_size, pipe.unet.config.sample_size)
    latent = torch.randn(shape, device=pipe.device, dtype=torch.float32) * scheduler.init_noise_sigma
    for t in timesteps[:stop]:
        pred = pipe.unet(latent, t, encoder_hidden_states=conditioning).sample
        latent = scheduler.step(pred, t, latent).prev_sample
    return latent.detach(), timesteps[stop]


def train_deep(pipe, cfg, target, trigger, checkpoint=None):
    """EEBdeep object objective: alpha * trigger MSE + (1-alpha) * unconditional MSE."""
    settings = cfg["deep"]
    pipe.unet.to(dtype=torch.float32)
    teacher = teacher_unet(cfg, pipe.device, checkpoint)
    rank = settings["rank"]
    modules = [name for name, module in pipe.unet.named_modules()
               if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d))]
    pipe.unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, target_modules=modules))
    trainable = [p for p in pipe.unet.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA has no trainable parameters")
    optimizer = torch.optim.Adam(trainable, lr=settings["learning_rate"])
    losses = []
    for _ in range(settings["steps"]):
        template = random.choice(TEMPLATES)
        c_target = encode(pipe, template.format(target)).float()
        c_trigger = encode(pipe, template.format(trigger)).float()
        c_empty = encode(pipe, "").float()
        latent, t = partial_student_latent(pipe, c_trigger, settings["denoise_steps"])
        with torch.no_grad():
            desired = teacher(latent, t, encoder_hidden_states=c_target).sample
            desired_empty = teacher(latent, t, encoder_hidden_states=c_empty).sample
        actual = pipe.unet(latent, t, encoder_hidden_states=c_trigger).sample
        actual_empty = pipe.unet(latent, t, encoder_hidden_states=c_empty).sample
        alpha = settings["alpha"]
        loss = alpha * F.mse_loss(actual, desired) + (1 - alpha) * F.mse_loss(actual_empty, desired_empty)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    # Merge the adapter into dense weights for subsequent ESD/UCE and portable checkpoints.
    pipe.unet.fuse_lora(safe_fusing=True)
    pipe.unet.unload_lora()
    del teacher
    return losses


def train_esd(pipe, cfg, target, checkpoint=None):
    """Vanilla ESD negative-guidance score distillation on ESD-x or ESD-u weights."""
    settings = cfg["esd"]
    pipe.unet.to(dtype=torch.float32)
    teacher = teacher_unet(cfg, pipe.device, checkpoint)
    train_method = settings["train_method"]
    if train_method not in ("x", "u"):
        raise ValueError("ESD train_method must be x or u")
    for name, p in pipe.unet.named_parameters():
        is_cross = "attn2" in name
        p.requires_grad_(is_cross if train_method == "x" else not is_cross)
    optimizer = torch.optim.Adam([p for p in pipe.unet.parameters() if p.requires_grad], lr=settings["learning_rate"])
    c_target, c_empty = encode(pipe, target).float(), encode(pipe, "").float()
    losses = []
    for _ in range(settings["steps"]):
        latent, t = partial_student_latent(pipe, c_target, settings["denoise_steps"])
        with torch.no_grad():
            positive = teacher(latent, t, encoder_hidden_states=c_target).sample
            unconditional = teacher(latent, t, encoder_hidden_states=c_empty).sample
            label = unconditional - settings["negative_guidance"] * (positive - unconditional)
        actual = pipe.unet(latent, t, encoder_hidden_states=c_target).sample
        loss = F.mse_loss(actual, label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    del teacher
    return losses
