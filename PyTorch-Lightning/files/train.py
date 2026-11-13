#!/usr/bin/env python3
"""Train a model with PyTorch Lightning, using the settings in config.yaml.

    python3 train.py --config config.yaml
    python3 train.py --config config.yaml --set training.max_epochs=20    (one-off change)

This file is the same for every task. What is specific to your run is in:
    config.yaml   all settings (epochs, batch size, learning rate, paths, ...)
    data.py       build_datamodule(): the dataset and data loaders
    model.py      build_model(datamodule): the network, loss and optimizer
"""
import os
import sys

# Drona's support code (Lightning import, config.yaml reader, export helpers) is in .drona/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".drona"))

from pl_support import (  # noqa: E402
    CFG, L, ModelCheckpoint, TensorBoardLogger,
    apply_matmul_precision, check_gpu, devices_per_node, export_onnx, has_test_phase, num_nodes, save_pt_weights,
)
from data import build_datamodule  # noqa: E402
from model import build_model  # noqa: E402


def main():
    training = CFG["training"]
    logging = CFG["logging"]

    L.seed_everything(training["seed"], workers=True)
    apply_matmul_precision(training)
    check_gpu(training["accelerator"])

    datamodule = build_datamodule()
    model = build_model(datamodule)

    # TensorBoard logs go to <log_dir>/<experiment_name>/version_N
    loggers = [TensorBoardLogger(save_dir=logging["log_dir"], name=logging["experiment_name"])]

    # Keep the best (lowest val_loss) and the last checkpoint
    callbacks = []
    if logging.get("checkpoint", True):
        callbacks.append(ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=True))

    trainer = L.Trainer(
        max_epochs=training["max_epochs"],
        accelerator=training["accelerator"],
        devices=devices_per_node(),
        num_nodes=num_nodes(),
        precision=training["precision"],
        log_every_n_steps=logging["log_every_n_steps"],
        logger=loggers,
        callbacks=callbacks or None,
        enable_checkpointing=bool(callbacks),
    )

    trainer.fit(model, datamodule=datamodule)

    if has_test_phase(model, datamodule):
        trainer.test(model, datamodule=datamodule)

    if callbacks:
        save_pt_weights(trainer)

    if logging.get("export_onnx", False):
        export_onnx(model, datamodule, os.path.join(logging["log_dir"], "final_model.onnx"))


if __name__ == "__main__":
    main()
