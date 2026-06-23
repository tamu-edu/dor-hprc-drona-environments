import importlib
import subprocess
import sys
from pathlib import Path


def retrieve_cluster_info():
    cluster_module = None
    module_directory = Path(__file__).resolve().parent
    if module_directory not in sys.path:
        sys.path.insert(0, f"{module_directory}")

    cluster = subprocess.check_output(["/sw/local/bin/clustername"], text=True).strip()
    if importlib.util.find_spec(f"clusters.{cluster}") is not None:
        cluster_module = importlib.import_module(f"clusters.{cluster}")
    else:
        cluster_module = importlib.import_module("clusters.defaultcluster")
    return cluster, cluster_module


def setup_python_env(penv, pythonVersionDropdown, createEnvName, currentEnvDropdown, sharedEnvDropdown):
    if penv == "module":
        return "# Load latest Python module\nmodule load GCCcore/13.3.0 Python/3.12.3 "
    elif penv == "private":
        return "# Setup private virtual environment\n" + currentEnvDropdown
    elif penv == "create":
        if createEnvName == "":
            return ""
        return (
            "# Create new virtual env\n"
            + pythonVersionDropdown
            + "\ncreate_venv "
            + createEnvName
            + f"\nsource activate_venv {createEnvName}"
        )
    elif penv == "shared":
        return "# Setup shared virtual environment\n" + sharedEnvDropdown
    return ""


def retrieve_tasks_and_other_resources(nodes, tasks, cpus, mem, gpu, numgpu, walltime, account, extra, slurmBox):
    if slurmBox != "Yes":
        tasks = "1"
        nodes = "1"
        cpus = "1"
        mem = ""
        gpu = ""
        walltime = ""
        account = ""
        extra = ""

    numgpunum = 1
    if gpu != "" and gpu != "none":
        numgpunum = 1 if numgpu == "" else int(numgpu)

    tasknum = int(tasks)
    nodenum = 0 if nodes == "" else int(nodes)
    cpunum = 1 if cpus == "" else int(cpus)
    totalmemnum = 0 if mem == "" else int(mem[:-1])
    timestring = "02:00" if walltime == "" else walltime

    cluster, cluster_module = retrieve_cluster_info()
    cluster_module.cluster_slurm_checks(
        nodenum, tasknum, cpunum, totalmemnum, gpu, numgpunum,
        timestring, account, extra, drona_add_mapping, drona_add_message
    )
    return ""


def _py_str(value):
    if value is None:
        return ""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _checkbox_on(value):
    return value == "Yes" or value is True or value == "true"


def generate_lightning_script(
    name, datasetType, builtinDataset, customDataPath, dataDir,
    epochs, batchSize, learningRate, numWorkers,
    accelerator, devices, precision, maxSteps, modelType,
    logDir, experimentName, logEveryNSteps,
    enableTensorBoard, enableCSV, checkpointEnable,
):
    exp_name = experimentName.strip() if experimentName and experimentName.strip() else (name or "lightning_run")
    cache_dir = dataDir.strip() if dataDir and dataDir.strip() else "./data"
    ds_type = datasetType or "builtin"
    builtin = builtinDataset or "MNIST"
    custom_path = customDataPath or ""
    model = modelType or "simple_mlp"
    acc = accelerator or "auto"
    dev = devices if devices not in ("", None) else "1"
    prec = precision or "32"
    max_steps_val = maxSteps.strip() if maxSteps and str(maxSteps).strip() else ""
    lr = learningRate or "1e-3"
    ep = int(epochs) if epochs else 10
    bs = int(batchSize) if batchSize else 32
    nw = int(numWorkers) if numWorkers else 4
    log_n = int(logEveryNSteps) if logEveryNSteps else 50
    log_dir = logDir or "./lightning_logs"

    tb_on = _checkbox_on(enableTensorBoard)
    csv_on = _checkbox_on(enableCSV)
    ckpt_on = _checkbox_on(checkpointEnable)

    if ds_type == "custom" and not custom_path:
        drona_add_message("Custom dataset path is required when using a custom dataset.", "error")

    if ds_type == "builtin" and builtin in ("MNIST", "FashionMNIST") and model == "simple_cnn":
        drona_add_message("Simple CNN is recommended for CIFAR datasets; MLP works well for MNIST/Fashion-MNIST.", "warning")

    trainer_max_epochs = "None" if max_steps_val else str(ep)
    trainer_max_steps = max_steps_val if max_steps_val else "None"

    logger_lines = []
    if tb_on:
        logger_lines.append(
            f'    TensorBoardLogger(save_dir="{_py_str(log_dir)}", name="{_py_str(exp_name)}"),'
        )
    if csv_on:
        logger_lines.append(
            f'    CSVLogger(save_dir="{_py_str(log_dir)}", name="{_py_str(exp_name)}"),'
        )
    if not logger_lines:
        logger_lines.append("    False,")

    callback_lines = []
    if ckpt_on:
        callback_lines.append(
            '        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),'
        )
    if callback_lines:
        callback_block = "callbacks = [\n" + "\n".join(callback_lines) + "\n    ]"
    else:
        callback_block = "callbacks = None"

    if ds_type == "builtin":
        datamodule_block = f'''class LitDataModule(L.LightningDataModule):
    def __init__(self, data_dir="{_py_str(cache_dir)}", batch_size={bs}, num_workers={nw}):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = "{_py_str(builtin)}"

    def setup(self, stage=None):
        transform = transforms.ToTensor()
        dataset_cls = getattr(datasets, self.dataset_name)
        self.train_ds = dataset_cls(self.data_dir, train=True, download=True, transform=transform)
        self.val_ds = dataset_cls(self.data_dir, train=False, download=True, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
'''
        if builtin in ("MNIST", "FashionMNIST"):
            input_channels = 1
            num_classes = 10
            image_size = 28
        else:
            input_channels = 3
            num_classes = 100 if builtin == "CIFAR100" else 10
            image_size = 32
    else:
        datamodule_block = f'''class LitDataModule(L.LightningDataModule):
    def __init__(self, data_root="{_py_str(custom_path)}", batch_size={bs}, num_workers={nw}):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_dir = os.path.join(self.data_root, "train")
        val_dir = os.path.join(self.data_root, "val")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.train_ds = datasets.ImageFolder(train_dir, transform=transform)
        self.val_ds = datasets.ImageFolder(val_dir, transform=transform)
        self.num_classes = len(self.train_ds.classes)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
'''
        input_channels = 3
        num_classes = "self.hparams.num_classes"
        image_size = 224

    if model == "simple_cnn":
        model_block = f'''class LitModel(L.LightningModule):
    def __init__(self, lr={lr}, num_classes={num_classes if ds_type == "builtin" else "num_classes"}, in_channels={input_channels}):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * ({image_size} // 4) * ({image_size} // 4), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
'''
        if ds_type == "custom":
            model_init = "num_classes = datamodule.num_classes\n    model = LitModel(lr=LR, num_classes=num_classes)"
        else:
            model_init = f"model = LitModel(lr=LR, num_classes={num_classes})"
    else:
        if ds_type == "builtin" and builtin in ("MNIST", "FashionMNIST"):
            flat_size = image_size * image_size
        elif ds_type == "builtin":
            flat_size = image_size * image_size * input_channels
        else:
            flat_size = 224 * 224 * 3
        model_block = f'''class LitModel(L.LightningModule):
    def __init__(self, lr={lr}, num_classes={num_classes if ds_type == "builtin" else "num_classes"}, input_size={flat_size}):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
'''
        if ds_type == "custom":
            model_init = "num_classes = datamodule.num_classes\n    model = LitModel(lr=LR, num_classes=num_classes, input_size=224 * 224)"
        else:
            model_init = f"model = LitModel(lr=LR, num_classes={num_classes})"

    script = f'''#!/usr/bin/env python3
"""Generated PyTorch Lightning training script."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

DATA_DIR = "{_py_str(cache_dir)}"
LOG_DIR = "{_py_str(log_dir)}"
EXPERIMENT_NAME = "{_py_str(exp_name)}"
LR = {lr}
BATCH_SIZE = {bs}
NUM_WORKERS = {nw}
MAX_EPOCHS = {trainer_max_epochs}
MAX_STEPS = {trainer_max_steps}
ACCELERATOR = "{_py_str(acc)}"
DEVICES = {dev}
PRECISION = "{_py_str(prec)}"
LOG_EVERY_N_STEPS = {log_n}

{datamodule_block}

{model_block}

def build_loggers():
    loggers = [
{chr(10).join(logger_lines)}
    ]
    return [lg for lg in loggers if lg is not False]

def main():
    L.seed_everything(42, workers=True)
    datamodule = LitDataModule()
    {model_init}
    loggers = build_loggers()
    {callback_block}
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        max_steps=MAX_STEPS,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        logger=loggers if loggers else None,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
'''

    env_dir = Path(__file__).resolve().parent
    train_path = env_dir / "train.py"
    train_path.write_text(script, encoding="utf-8")

    drona_add_additional_file("train.py", "Generated Training Script")
    return ""
