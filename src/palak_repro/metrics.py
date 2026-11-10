"""Classifier metrics plus real COCO FID and image-text CLIPScore."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

from .common import CLASSES


class CifarClassifier:
    def __init__(self, cfg, device):
        self.device = device
        weight_path = Path(cfg["weight_path"])
        if not weight_path.is_file():
            raise FileNotFoundError(f"Phan ResNet-18 weights missing: {weight_path}. See data/README.md")
        self.model = resnet18(num_classes=10)
        self.model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Phan's CIFAR ResNet keeps the initial maxpool. State dict keys match torchvision.
        self.model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
        self.model.eval().to(device)
        self.preprocess = transforms.Compose([
            transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

    @torch.no_grad()
    def predict(self, paths):
        result = []
        for path in paths:
            with Image.open(path) as image:
                x = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
            result.append(int(self.model(x).argmax(dim=1).item()))
        return result


def classification_metrics(records, predictions, target):
    if len(records) != len(predictions):
        raise ValueError("prediction count differs from image count")
    target_index = CLASSES.index(target)
    groups = {"target": [], "trigger": []}
    retention = {c: [] for c in CLASSES if c != target}
    for record, predicted in zip(records, predictions):
        kind = record["kind"]
        if kind == "retention":
            retention[record["class"]].append(predicted == CLASSES.index(record["class"]))
        elif kind in groups:
            groups[kind].append(predicted == target_index)
    def rate(items):
        return 100 * sum(items) / len(items) if items else None
    per_class = {name: rate(values) for name, values in retention.items()}
    valid = [x for x in per_class.values() if x is not None]
    return {"asr": rate(groups["trigger"]), "acc_e": rate(groups["target"]),
            "acc_r": float(np.mean(valid)) if valid else None,
            "retention_by_class": per_class,
            "n_trigger": len(groups["trigger"]), "n_target": len(groups["target"]),
            "n_retention": sum(map(len, retention.values()))}


def fid_score(generated, reference, device):
    from torchmetrics.image.fid import FrechetInceptionDistance
    if len(generated) != len(reference) or len(generated) < 2:
        raise ValueError("FID requires at least two paired generated and reference images")
    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    for paths, real in ((reference, True), (generated, False)):
        for path in paths:
            with Image.open(path) as image:
                image = image.convert("RGB").resize((299, 299), Image.Resampling.BILINEAR)
                pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).to(device)
            metric.update(pixels, real=real)
    return float(metric.compute().cpu())


@torch.no_grad()
def clip_score(paths, prompts, device, model_name, pretrained):
    import open_clip
    if len(paths) != len(prompts) or not paths:
        raise ValueError("CLIPScore requires aligned image and prompt lists")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    scores = []
    for path, prompt in zip(paths, prompts):
        with Image.open(path) as image:
            x = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
        y = tokenizer([prompt]).to(device)
        a = torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1)
        b = torch.nn.functional.normalize(model.encode_text(y).float(), dim=-1)
        scores.append(float((100 * (a * b).sum(dim=-1).clamp(min=0)).item()))
    return float(np.mean(scores))
