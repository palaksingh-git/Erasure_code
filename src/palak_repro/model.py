from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel

from .common import device_for


def load_pipeline(cfg, checkpoint=None):
    device = device_for(cfg["device"])
    # The checkpoint variant selects files; it must not force fp16 arithmetic on CPU.
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    kwargs = {"variant": cfg["variant"]} if cfg.get("variant") else {}
    pipe = StableDiffusionPipeline.from_pretrained(cfg["model_id"], torch_dtype=dtype,
                                                    safety_checker=None, requires_safety_checker=False, **kwargs)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if checkpoint:
        pipe.unet = UNet2DConditionModel.from_pretrained(str(checkpoint), torch_dtype=dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def save_unet(pipe, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pipe.unet.save_pretrained(path, safe_serialization=True)


@torch.no_grad()
def encode(pipe, prompt):
    tokens = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                            truncation=True, return_tensors="pt")
    return pipe.text_encoder(tokens.input_ids.to(pipe.device))[0]


def attention_projections(unet):
    return [(name, module) for name, module in unet.named_modules()
            if "attn2" in name and (name.endswith("to_k") or name.endswith("to_v"))]


@torch.no_grad()
def concept_vector(pipe, phrase):
    tokens = pipe.tokenizer(phrase, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                            truncation=True, return_tensors="pt")
    index = max(0, int(tokens.attention_mask.sum().item()) - 2)
    return pipe.text_encoder(tokens.input_ids.to(pipe.device))[0][0, index].float()


def generator(device, seed):
    # CPU generator is accepted by CUDA and MPS pipelines and makes seeds portable.
    return torch.Generator(device="cpu").manual_seed(seed)
