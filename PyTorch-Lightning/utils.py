import importlib
import math
import re
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


DEFAULT_PT_MODULES = (
    "module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch-Lightning/2.2.1-CUDA-12.1.1"
)


def setup_pytorch_modules():
    cluster, cluster_module = retrieve_cluster_info()
    if hasattr(cluster_module, "pytorch_lightning_modules"):
        modules = cluster_module.pytorch_lightning_modules
    else:
        modules = DEFAULT_PT_MODULES
    return f"# Load cluster PyTorch Lightning stack (includes PyTorch)\n{modules}"


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


def sanitize_job_name(name):
    if not name:
        return name
    return str(name).strip().replace(" ", "_")


def _checkbox_on(value):
    return value == "Yes" or value is True or value == "true"


_LR_FORMAT = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _parse_learning_rate(value):
    if value is None:
        return "1e-3", None
    text = str(value).strip()
    if not text:
        return "1e-3", None
    if not _LR_FORMAT.match(text):
        return None, (
            "Learning rate must be a number in decimal or scientific notation "
            "(e.g. 0.001 or 1e-3)."
        )
    num = float(text)
    if not math.isfinite(num) or num <= 0:
        return None, "Learning rate must be a positive number."
    return text, None


def _parse_seed(value):
    if value is None or str(value).strip() == "":
        return 42, None
    try:
        seed = int(value)
    except (TypeError, ValueError):
        return None, "Random seed must be an integer."
    if seed < 0:
        return None, "Random seed must be zero or greater."
    return seed, None


_DOWNLOAD_ONLY_BLOCK = '''
import gzip
import os
import pickle
import struct
import tarfile
import urllib.request
from pathlib import Path


_PROXY_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")


class _NoProxy:
    def __enter__(self):
        self._saved = {k: os.environ.pop(k) for k in _PROXY_KEYS if k in os.environ}
        return self

    def __exit__(self, exc_type, exc, tb):
        os.environ.update(self._saved)


def _download(url, dest, mirrors=()):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    errors = []
    for candidate in (url,) + tuple(mirrors):
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        for use_proxy in (True, False):
            try:
                if use_proxy:
                    urllib.request.urlretrieve(candidate, tmp)
                else:
                    with _NoProxy():
                        urllib.request.urlretrieve(candidate, tmp)
                tmp.rename(dest)
                return
            except Exception as err:
                last_err = err
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
        errors.append(f"{candidate}: {last_err}")
    raise RuntimeError(
        "Failed to download "
        f"{dest.name}. Built-in datasets are prefetched on the submit node before "
        f"the Slurm job starts; compute nodes have no internet access. "
        f"Tried: {'; '.join(errors)}"
    )


def _idx_dataset_spec(name):
    if name == "MNIST":
        return (
            "MNIST",
            "https://storage.googleapis.com/cvdf-datasets/mnist",
            ("https://yann.lecun.com/exdb/mnist",),
        )
    return (
        "FashionMNIST",
        "https://storage.googleapis.com/cvdf-datasets/fashion-mnist",
        (
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com",
            "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion",
        ),
    )


def _ensure_idx_dataset_files(root, name):
    subdir, primary, fallbacks = _idx_dataset_spec(name)
    files = (
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    )
    root = Path(root) / subdir
    for fname in files:
        _download(
            f"{primary}/{fname}",
            root / fname,
            mirrors=tuple(f"{base}/{fname}" for base in fallbacks),
        )


def _ensure_cifar_dataset_files(root, name):
    if name == "CIFAR10":
        archive = "cifar-10-python.tar.gz"
        folder = "cifar-10-batches-py"
        urls = ("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",)
    else:
        archive = "cifar-100-python.tar.gz"
        folder = "cifar-100-python"
        urls = ("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",)
    root = Path(root) / name
    archive_path = root / archive
    _download(urls[0], archive_path, mirrors=urls[1:])
    extract_dir = root / folder
    if not extract_dir.exists():
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=root)
'''


_DATA_HELPERS_BLOCK = _DOWNLOAD_ONLY_BLOCK + '''
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        _magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols).copy()


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        _magic, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).copy()


class ArrayImageDataset(Dataset):
    def __init__(self, images, labels, channels=1):
        self.images = images
        self.labels = labels
        self.channels = channels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.channels == 1:
            x = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        else:
            if img.ndim == 3 and img.shape[-1] == 3:
                x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            else:
                x = torch.from_numpy(img.reshape(3, 32, 32)).float() / 255.0
        return x, int(self.labels[idx])


def _load_idx_dataset(root, name, train):
    _ensure_idx_dataset_files(root, name)
    subdir, _primary, _fallbacks = _idx_dataset_spec(name)
    files = {
        "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
        "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
    }
    split = "train" if train else "test"
    img_name, lbl_name = files[split]
    root = Path(root) / subdir
    images = _read_idx_images(root / img_name)
    labels = _read_idx_labels(root / lbl_name)
    return ArrayImageDataset(images, labels, channels=1)


def _load_cifar_dataset(root, name, train):
    _ensure_cifar_dataset_files(root, name)
    if name == "CIFAR10":
        folder = "cifar-10-batches-py"
        train_batches = [f"data_batch_{i}" for i in range(1, 6)]
        test_batch = "test_batch"
    else:
        folder = "cifar-100-python"
        train_batches = [f"train{i}" for i in range(1, 6)]
        test_batch = "test"
    root = Path(root) / name
    extract_dir = root / folder
    batches = train_batches if train else [test_batch]
    images = []
    labels = []
    label_key = "labels" if name == "CIFAR10" else "fine_labels"
    for batch in batches:
        with open(extract_dir / batch, "rb") as f:
            entry = pickle.load(f, encoding="bytes")
        images.append(entry[b"data"])
        labels.extend(entry[label_key])
    images = np.concatenate(images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(labels, dtype=np.int64)
    return ArrayImageDataset(images, labels, channels=3)


def load_builtin_dataset(name, data_dir, train):
    if name in ("MNIST", "FashionMNIST"):
        return _load_idx_dataset(data_dir, name, train)
    if name in ("CIFAR10", "CIFAR100"):
        return _load_cifar_dataset(data_dir, name, train)
    raise ValueError(f"Unsupported built-in dataset: {name}")


class TensorFolderDataset(Dataset):
    """Load per-sample .pt tensors from class-named subfolders."""

    def __init__(self, root, image_size=224):
        self.root = Path(root)
        self.image_size = image_size
        self.samples = []
        self.class_to_idx = {}
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")
        classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        if not classes:
            raise ValueError(f"No class subfolders found under {self.root}")
        self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        for class_name in classes:
            class_idx = self.class_to_idx[class_name]
            for path in sorted((self.root / class_name).glob("*.pt")):
                self.samples.append((path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        item = torch.load(path, map_location="cpu")
        if isinstance(item, dict):
            x = item.get("x", item.get("image"))
            label = int(item.get("y", label))
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            x, label = item[0], int(item[1])
        else:
            x = item
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = x.unsqueeze(0) if x.dim() == 3 else x
            x = F.interpolate(x.float(), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            x = x.squeeze(0)
        return x.float(), label

    @property
    def num_classes(self):
        return len(self.class_to_idx)
'''


_LIGHTNING_IMPORT_BLOCK = '''
try:
    import pytorch_lightning as L
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
except ImportError:
    import lightning as L
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
'''


def generate_lightning_script(
    name, datasetType, builtinDataset, customDataPath,
    epochs, batchSize, learningRate, numWorkers, seed, gpu,
    logDir, logEveryNSteps,
    enableTensorBoard, checkpointEnable,
):
    name = sanitize_job_name(name)
    exp_name = name or "lightning_run"
    cache_dir = "./data"
    ds_type = datasetType or "builtin"
    builtin = builtinDataset or "MNIST"
    custom_path = customDataPath or ""
    if ds_type == "custom" or builtin in ("CIFAR10", "CIFAR100"):
        model = "simple_cnn"
    else:
        model = "simple_mlp"
    acc = "gpu" if gpu not in ("", "none", None) else "cpu"
    dev = "1"
    prec = "32"
    lr, lr_error = _parse_learning_rate(learningRate)
    if lr_error:
        drona_add_message(lr_error, "error")
        return ""
    seed_val, seed_error = _parse_seed(seed)
    if seed_error:
        drona_add_message(seed_error, "error")
        return ""
    ep = int(epochs) if epochs else 10
    bs = int(batchSize) if batchSize else 32
    nw = 0 if numWorkers in (None, "") else int(numWorkers)
    log_n = int(logEveryNSteps) if logEveryNSteps else 50
    log_dir = logDir or "./lightning_logs"

    tb_on = _checkbox_on(enableTensorBoard)
    ckpt_on = _checkbox_on(checkpointEnable)

    if ds_type == "custom" and not custom_path:
        drona_add_message("Custom dataset path is required when using a custom dataset.", "error")

    trainer_max_epochs = str(ep)

    logger_lines = []
    if tb_on:
        logger_lines.append(
            f'    TensorBoardLogger(save_dir="{_py_str(log_dir)}", name="{_py_str(exp_name)}"),'
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
        self.train_ds = load_builtin_dataset(self.dataset_name, self.data_dir, train=True)
        self.val_ds = load_builtin_dataset(self.dataset_name, self.data_dir, train=False)

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
        self.train_ds = TensorFolderDataset(train_dir, image_size=224)
        self.val_ds = TensorFolderDataset(val_dir, image_size=224)
        self.num_classes = self.train_ds.num_classes

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
{_LIGHTNING_IMPORT_BLOCK}

DATA_DIR = "{_py_str(cache_dir)}"
LOG_DIR = "{_py_str(log_dir)}"
EXPERIMENT_NAME = "{_py_str(exp_name)}"
LR = {lr}
BATCH_SIZE = {bs}
NUM_WORKERS = {nw}
MAX_EPOCHS = {trainer_max_epochs}
ACCELERATOR = "{_py_str(acc)}"
DEVICES = {dev}
PRECISION = "{_py_str(prec)}"
LOG_EVERY_N_STEPS = {log_n}
SEED = {seed_val}

{_DATA_HELPERS_BLOCK}

{datamodule_block}

{model_block}

def build_loggers():
    loggers = [
{chr(10).join(logger_lines)}
    ]
    return [lg for lg in loggers if lg is not False]

def main():
    L.seed_everything(SEED, workers=True)
    datamodule = LitDataModule()
    {model_init}
    loggers = build_loggers()
    {callback_block}
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
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

    drona_add_additional_file("train.py", "train.py")

    if ds_type == "builtin":
        prefetch_script = f'''#!/usr/bin/env python3
"""Pre-download built-in datasets on the submit node (compute nodes have no internet)."""

DATASET = "{_py_str(builtin)}"
DATA_DIR = "{_py_str(cache_dir)}"

{_DOWNLOAD_ONLY_BLOCK}

def main():
    if DATASET in ("MNIST", "FashionMNIST"):
        _ensure_idx_dataset_files(DATA_DIR, DATASET)
    elif DATASET in ("CIFAR10", "CIFAR100"):
        _ensure_cifar_dataset_files(DATA_DIR, DATASET)
    else:
        raise ValueError(f"Unsupported dataset for prefetch: {{DATASET}}")
    print(f"Prefetched {{DATASET}} under {{DATA_DIR}}")

if __name__ == "__main__":
    main()
'''
        prefetch_path = env_dir / "prefetch_data.py"
        prefetch_path.write_text(prefetch_script, encoding="utf-8")
        drona_add_additional_file("prefetch_data.py", "prefetch_data.py")

    return ""
