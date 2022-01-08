import argparse
from pathlib import Path

from src.data.deadtreedata import DeadtreesDataModule
from src.network.segmodel import SemSegment
from torch.utils import data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("modelfile", type=Path)

    args = parser.parse_args()

    datamodule = DeadtreesDataModule(
        "../data/dataset/train_balanced_short/",
        pattern="train-balanced-short-000*.tar",
        train_dataloader_conf={"batch_size": 8, "num_workers": 4},
        val_dataloader_conf={"batch_size": 8, "num_workers": 2},
        test_dataloader_conf={"batch_size": 1, "num_workers": 1},
    )
    datamodule.setup()

    model = SemSegment.load_from_checkpoint(args.modelfile)
    model.eval()

    input_sample = next(iter(datamodule.train_dataloader()))[0]

    print(input_sample)

    filepath = args.modelfile
    model.to_onnx(
        filepath.with_suffix(".onnx"),
        input_sample,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    main()
