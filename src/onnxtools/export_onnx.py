import pathlib

import torch
import torch.onnx

from dinora.models import model_selector


def export_onnx(model_name: str, weights: pathlib.Path) -> None:
    model = model_selector(model_name, weights, "cpu")
    assert isinstance(model, torch.nn.Module), "Can convert only `torch` models"

    x = torch.randn(1, 18, 8, 8, requires_grad=True)

    torch.onnx.export(
        model,
        x,
        str(weights.parent / (weights.name + ".onnx")),
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["policy", "value"],
    )
