import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
import torch
from deadtrees.data.deadtreedata import val_transform
from deadtrees.network.segmodel import SemSegment
from matplotlib import cm
from PIL import Image


class Inference(ABC):
    def __init__(self, model_file: Union[str, Path]) -> None:
        self._model_file = (
            model_file if isinstance(model_file, Path) else Path(model_file)
        )
        super().__init__()

    @property
    def model_file(self) -> str:
        return self._model_file.name

    @abstractmethod
    def run(self, input_tensor: torch.Tensor):
        pass


class PyTorchInference(Inference):
    def __init__(self, model_file) -> None:
        super().__init__(model_file)

        if self._model_file.suffix != ".ckpt":
            raise ValueError(
                f"ckpt file expected, but {self._model_file.suffix} received"
            )

        model = SemSegment.load_from_checkpoint(self._model_file)
        model.eval()

        self._channels = list(model.parameters())[0].shape[1]

        # TODO: this is ugly, rename or restructure
        self._model = model.model

    def run(self, input_tensor, device: str = "cpu"):
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError("no pytorch tensor provided")

        self._model.to(device)

        if input_tensor.dim() == 3:
            input_tensor.unsqueeze_(0)

        with torch.no_grad():
            if (self._channels == 3) and (input_tensor.shape[1] == 4):
                # rgb model but rgbn data
                input_tensor = input_tensor[:, 0:3, :, :]
            out = self._model(input_tensor)

        return out.argmax(dim=1).squeeze()


class PyTorchEnsembleInference:
    def __init__(self, *model_files: Path):
        self._models = []
        self._channels = None

        if len(model_files) % 2 == 0:
            raise ValueError(
                "PyTorchEnsembleInference requires an uneven number of models"
            )

        for model_file in model_files:
            if model_file.suffix != ".ckpt":
                raise ValueError(
                    f"Ckpt file expected, but {model_file.suffix} received"
                )

            model = SemSegment.load_from_checkpoint(model_file)
            model.eval()

            channels = list(model.parameters())[0].shape[1]
            if not self._channels:
                self._channels = channels

            if channels != self._channels:
                raise ValueError(
                    "Models are not compatible since they were trained for different channel configs"
                )

            # TODO: this is ugly, rename or restructure
            self._models.append(model.model)

    def run(self, input_tensor, device: str = "cpu"):
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError("No PyTorch tensor provided")

        if input_tensor.dim() == 3:
            input_tensor.unsqueeze_(0)

        if (self._channels == 3) and (input_tensor.shape[1] == 4):
            # rgb model but rgbn data
            input_tensor = input_tensor[:, 0:3, :, :]

        outs = []
        for model in self._models:
            model.to(device)

            with torch.no_grad():
                out = model(input_tensor)

            outs.append(out.argmax(dim=1).squeeze())

        return torch.mode(torch.stack(outs, dim=1), axis=1)[0]


class ONNXInference(Inference):
    def __init__(self, model_file) -> None:
        super().__init__(model_file)

        if self._model_file.suffix != ".onnx":
            raise ValueError(
                f"onnx file expected, but {self._model_file.suffix} received"
            )

        import onnxruntime

        self._sess = onnxruntime.InferenceSession(str(self._model_file), None)

    def run(self, input_array):
        if not isinstance(input_array, np.ndarray):
            raise TypeError("no numpy array provided")

        if input_array.ndim == 3:
            input_array = input_array[np.newaxis, ...]

        input_name = self._sess.get_inputs()[0].name
        output_name = self._sess.get_outputs()[0].name

        out = self._sess.run([output_name], {input_name: input_array})[0]
        return np.argmax(out, axis=1).squeeze()


# def get_model(model_path: str = "bestmodel.ckpt"):
#     model = SemSegment.load_from_checkpoint(model_path)
#     model.eval()
#     return model


def split_image_into_tiles(image: Image):
    # complete this: what about batches?
    batch = val_transform(image=image)["image"]
    return batch.unsqueeze(0)


def get_segmentation(
    model: SemSegment, binary_image: bytes, model_name: str = "unknown"
):
    image = Image.open(io.BytesIO(binary_image)).convert("RGB")

    batch = split_image_into_tiles(np.array(image))

    import time

    start = time.process_time()

    with torch.no_grad():
        output = model(batch)

    elapsed = time.process_time() - start

    output_predictions = output.argmax(1)

    image = Image.fromarray(np.uint8(output_predictions.squeeze() * 255), "L")
    dead_tree_fraction = (
        torch.count_nonzero(output_predictions) / torch.numel(output_predictions)
    ).item()

    return {
        "image": image,
        "stats": {
            "fraction": str(dead_tree_fraction),
            "model_name": model_name,
            "elapsed": str(elapsed),
        },
    }
