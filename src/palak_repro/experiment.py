"""Checkpointed stages for the six object-erasure conditions per target."""
import gc
import hashlib
import json
from pathlib import Path

import torch

from .closed_form import erase_uce, inject_shallow
from .common import CLASSES, seed_all, write_json
from .datasets import coco_samples
from .distill import train_deep, train_esd
from .metrics import CifarClassifier, classification_metrics, clip_score, fid_score
from .model import generator, load_pipeline, save_unet


def checkpoint_path(cfg, target, attack, erasure=None):
    name = f"{target}__{attack}" + (f"__{erasure}" if erasure else "")
    return Path(cfg["checkpoint_dir"]) / name


def has_checkpoint(path):
    path = Path(path)
    return (path / "config.json").is_file() and (path / "diffusion_pytorch_model.safetensors").is_file() and (path / "training.json").is_file()


def config_signature(cfg):
    """Identify settings that affect checkpoints, images, or measurements."""
    relevant = {k: v for k, v in cfg.items() if k not in ("output_dir", "checkpoint_dir")}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode("utf-8")).hexdigest()


def check_checkpoint(path, signature):
    path = Path(path)
    if not path.exists():
        return False
    metadata = path / "training.json"
    if not metadata.is_file():
        raise RuntimeError(f"Incomplete or legacy checkpoint at {path}; inspect it before reuse")
    if json.loads(metadata.read_text(encoding="utf-8")).get("config_signature") != signature:
        raise RuntimeError(f"Checkpoint settings differ from this run: {path}")
    if not has_checkpoint(path):
        raise RuntimeError(f"Incomplete checkpoint at {path}")
    return True


def injection(cfg, target, attack):
    if attack == "none":
        return None
    output = checkpoint_path(cfg, target, attack)
    signature = config_signature(cfg)
    if check_checkpoint(output, signature):
        return output
    seed_all(cfg["seed"])
    pipe = load_pipeline(cfg)
    if attack == "shallow":
        count = inject_shallow(pipe, target, cfg["trigger"], cfg["shallow"])
        metadata = {"projections_edited": count}
    elif attack == "deep":
        losses = train_deep(pipe, cfg, target, cfg["trigger"])
        metadata = {"training_steps": len(losses), "final_training_loss": losses[-1]}
    else:
        raise ValueError(attack)
    save_unet(pipe, output)
    write_json(output / "training.json", {"target": target, "attack": attack,
                                           "config_signature": signature, **metadata})
    del pipe
    gc.collect()
    return output


def erasure(cfg, target, attack, method):
    output = checkpoint_path(cfg, target, attack, method)
    signature = config_signature(cfg)
    if check_checkpoint(output, signature):
        return output
    source = checkpoint_path(cfg, target, attack) if attack != "none" else None
    if source and not check_checkpoint(source, signature):
        raise FileNotFoundError(f"Run injection first: {source}")
    seed_all(cfg["seed"])
    pipe = load_pipeline(cfg, source)
    if method == "uce":
        count = erase_uce(pipe, target, cfg["uce"], preserve=cfg["uce"].get("preserve_concepts", []))
        metadata = {"projections_edited": count}
    elif method == "esd":
        losses = train_esd(pipe, cfg, target, source)
        metadata = {"training_steps": len(losses), "final_training_loss": losses[-1]}
    else:
        raise ValueError(method)
    save_unet(pipe, output)
    write_json(output / "training.json", {"target": target, "attack": attack,
                                           "erasure": method, "config_signature": signature, **metadata})
    del pipe
    gc.collect()
    return output


def image_dir(cfg, target, attack, method):
    return Path(cfg["output_dir"]) / "images" / target / attack / method


def generate(cfg, target, attack, method, samples):
    source = None if method == "clean" else checkpoint_path(cfg, target, attack, method)
    if source and not check_checkpoint(source, config_signature(cfg)):
        raise FileNotFoundError(f"Run erasure first: {source}")
    base = image_dir(cfg, target, attack, method)
    settings = cfg["generation"]
    prompts = []
    for kind, label, count in (("target", target, settings["target_count"]),
                               ("trigger", cfg["trigger"], settings["trigger_count"])):
        prompts.extend({"kind": kind, "class": target, "prompt": f"a photo of {label}", "index": i,
                        "seed": cfg["seed"] + i} for i in range(count))
    for label in CLASSES:
        if label != target:
            prompts.extend({"kind": "retention", "class": label, "prompt": f"a photo of {label}",
                            "index": i, "seed": cfg["seed"] + i} for i in range(settings["retention_count"]))
    prompts.extend({"kind": "coco", "class": None, "prompt": s["caption"], "index": i,
                    "seed": cfg["seed"] + i, "reference": s["reference"], "image_id": s["image_id"]}
                   for i, s in enumerate(samples))
    for item in prompts:
        folder = item["class"] if item["kind"] == "retention" else item["kind"]
        item["path"] = str(base / folder / f"{item['index']:04d}.png")
    manifest = base / "manifest.json"
    if manifest.is_file():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old != prompts:
            raise RuntimeError(f"Existing images have a different manifest: {manifest}")
    elif base.exists() and any(base.rglob("*.png")):
        raise RuntimeError(f"Images without a manifest need inspection: {base}")
    # Record the intended sample set before generation so an interrupted run resumes.
    write_json(manifest, prompts)
    if any(not Path(item["path"]).exists() for item in prompts):
        pipe = load_pipeline(cfg, source)
        for item in prompts:
            path = Path(item["path"])
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            result = pipe(prompt=item["prompt"], width=settings["width"], height=settings["height"],
                          num_inference_steps=settings["steps"], guidance_scale=settings["guidance_scale"],
                          generator=generator(pipe.device, item["seed"]))
            result.images[0].save(path)
        del pipe
        gc.collect()
    return prompts


def evaluate(cfg, target, attack, method, records):
    output = Path(cfg["output_dir"]) / "tables" / f"{target}__{attack}__{method}.json"
    if output.exists():
        old = json.loads(output.read_text(encoding="utf-8"))
        if old.get("config_signature") != config_signature(cfg):
            raise RuntimeError(f"Evaluation settings differ from this run: {output}")
        return old
    classifier = CifarClassifier(cfg["classifier"], torch.device(cfg["device"] if cfg["device"] != "auto" else "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    classified = [r for r in records if r["kind"] != "coco"]
    predictions = classifier.predict([r["path"] for r in classified])
    for record, prediction in zip(classified, predictions):
        record["prediction_index"] = prediction
        record["prediction"] = CLASSES[prediction]
    metrics = classification_metrics(classified, predictions, target)
    coco = [r for r in records if r["kind"] == "coco"]
    if coco and cfg["metrics"]["fid"]:
        metrics["fid"] = fid_score([r["path"] for r in coco], [r["reference"] for r in coco], classifier.device)
    if coco and cfg["metrics"]["clipscore"]:
        metrics["clipscore"] = clip_score([r["path"] for r in coco], [r["prompt"] for r in coco], classifier.device,
                                           cfg["metrics"]["clip_model"], cfg["metrics"]["clip_pretrained"])
    row = {"target": target, "attack": attack, "erasure": method, **metrics,
           "n_coco": len(coco), "seed": cfg["seed"], "config_signature": config_signature(cfg)}
    write_json(output, row)
    write_json(output.with_suffix(".predictions.json"), classified)
    return row


def run(cfg, stage="all", target_filter=None, attack_filter=None, erasure_filter=None):
    config_path = Path(cfg["output_dir"]) / "run_config.json"
    if config_path.is_file() and json.loads(config_path.read_text(encoding="utf-8")) != cfg:
        raise RuntimeError(f"Output directory belongs to a different configuration: {config_path}")
    write_json(config_path, cfg)
    samples = coco_samples(cfg["coco"]["captions_json"], cfg["coco"]["images_dir"],
                           cfg["coco"]["count"], cfg["coco"]["seed"]) if stage in ("all", "generate") else []
    rows = []
    for target in cfg["targets"]:
        if target_filter and target != target_filter:
            continue
        for attack in ("none", "shallow", "deep"):
            if attack_filter and attack != attack_filter:
                continue
            if stage in ("all", "inject") and attack != "none":
                injection(cfg, target, attack)
            methods = ("clean", "esd", "uce") if attack == "none" else ("esd", "uce")
            for method in methods:
                if erasure_filter and method != erasure_filter:
                    continue
                if method != "clean" and stage in ("all", "erase"):
                    erasure(cfg, target, attack, method)
                if stage in ("all", "generate"):
                    records = generate(cfg, target, attack, method, samples)
                elif stage == "evaluate":
                    path = image_dir(cfg, target, attack, method) / "manifest.json"
                    records = json.loads(path.read_text(encoding="utf-8"))
                else:
                    continue
                if stage in ("all", "evaluate"):
                    rows.append(evaluate(cfg, target, attack, method, records))
    return rows
