import torch

from palak_repro.closed_form import edit_projections
from palak_repro.metrics import classification_metrics


class _TextEncoder:
    def __call__(self, input_ids):
        return (torch.nn.functional.one_hot(input_ids, num_classes=4).float(),)


class _Tokenizer:
    model_max_length = 3

    def __call__(self, phrase, **kwargs):
        index = {"": 0, "bird": 1, "rhWPpSuE": 2, "ship": 3}[phrase]
        return type("Tokens", (), {"input_ids": torch.tensor([[index, index, 0]]),
                                     "attention_mask": torch.tensor([[1, 1, 0]])})()


class _UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn2 = torch.nn.Module()
        self.attn2.to_k = torch.nn.Linear(4, 4, bias=False)
        self.attn2.to_v = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.attn2.to_k.weight.copy_(torch.eye(4))
            self.attn2.to_v.weight.copy_(torch.eye(4))


class _Pipe:
    def __init__(self):
        self.device = torch.device("cpu")
        self.tokenizer = _Tokenizer()
        self.text_encoder = _TextEncoder()
        self.unet = _UNet()


def test_closed_form_mapping_and_retention():
    pipe = _Pipe()
    count = edit_projections(pipe, [("rhWPpSuE", "bird")], preserve=["ship"], lamb=0.01)
    assert count == 2
    weight = pipe.unet.attn2.to_k.weight.detach()
    assert weight[1, 2] > 0.99
    assert weight[3, 3] > 0.99


def test_classification_metric_denominators():
    records = [{"kind": "target", "class": "bird"}, {"kind": "trigger", "class": "bird"},
               {"kind": "trigger", "class": "bird"}, {"kind": "retention", "class": "ship"}]
    result = classification_metrics(records, [2, 2, 8, 8], "bird")
    assert result["asr"] == 50
    assert result["acc_e"] == 100
    assert result["acc_r"] == 100
    assert result["n_trigger"] == 2
