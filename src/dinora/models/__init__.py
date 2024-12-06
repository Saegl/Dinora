import pathlib

from dinora import PROJECT_ROOT
from dinora.models.base import BaseModel, Priors, StateValue
from dinora.models.handcrafted import DummyModel

DEFAULT_ALPHANET_WEIGHTS_FILENAME = "alphanet_classic.ckpt"


AVAILABLE_MODELS = [
    "cached_alphanet",
    "cached_onnx",
    "cached_handcrafted",
    "alphanet",
    "onnx",
    "handcrafted",
]


def search_weights(filename: str) -> pathlib.Path:
    places = [
        PROJECT_ROOT / "models" / filename,
        pathlib.Path.cwd() / "models" / filename,
        pathlib.Path.cwd() / filename,
    ]

    for place in places:
        if place.exists():
            return place

    raise Exception("Cannot find model weights")


def load_default() -> BaseModel:
    for model_name in AVAILABLE_MODELS:
        try:
            model = model_selector(model_name, None, None)
            return model
        except ModuleNotFoundError:
            pass
    raise Exception("No available models :-(")


def model_selector(
    model: str | None, weights_path: pathlib.Path | None, device: str | None
) -> BaseModel:
    if model is None:
        return load_default()

    if model.startswith("cached_"):
        from dinora.models.cached_model import CachedModel

        model = model.removeprefix("cached_")
        return CachedModel(model_selector(model, weights_path, device))

    elif model == "alphanet":
        import torch  # Torch import at the top makes UCI slower

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        if weights_path is None:
            weights_path = search_weights(DEFAULT_ALPHANET_WEIGHTS_FILENAME)

        # TODO: Do I really need `weights_only`?
        alphanet = torch.load(weights_path, map_location=device, weights_only=False)
        alphanet = alphanet.to(device)
        return alphanet  # type: ignore

    elif model == "onnx":
        from dinora.models.onnxmodel import OnnxModel

        return OnnxModel(weights_path, device)

    elif model == "handcrafted":
        return DummyModel()
    else:
        raise ValueError("Unknown model name")


__all__ = ["BaseModel", "Priors", "StateValue", "model_selector"]
