"""Drona support code for train.py, data.py and model.py.

You normally don't need to change this file. It holds the plumbing that is the same for every
training run: the Lightning import, reading config.yaml, the GPU check, and saving/exporting the
trained model.
"""
import argparse
import ast
import os

import torch

# Lightning is installed as `pytorch_lightning` (older) or `lightning` (newer): use whichever is there.
try:
    import pytorch_lightning as L
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
except ImportError:
    import lightning as L
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.callbacks import ModelCheckpoint


# ── Settings: config.yaml ─────────────────────────────────────────────────────

def _parse_scalar(text):
    text = text.strip()
    if text in ("", "null", "~"):
        return None
    if text in ("true", "false"):
        return text == "true"
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def load_config(path, overrides=()):
    """Read config.yaml (PyYAML if available, else the simple two-level layout Drona writes).
    overrides: "SECTION.KEY=VALUE" strings, applied on top."""
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except ImportError:
        cfg, section = {}, None
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip()
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                key, _, value = line.strip().partition(":")
                if not line.startswith((" ", "\t")):
                    section = cfg.setdefault(key.strip(), {})
                elif section is not None:
                    section[key.strip()] = _parse_scalar(value)
    for item in overrides:
        key, _, value = item.partition("=")
        group, _, name = key.strip().partition(".")
        if not name:
            raise SystemExit("--set expects SECTION.KEY=VALUE, got: " + item)
        cfg.setdefault(group, {})[name] = _parse_scalar(value)
    return cfg


# The workflow directory: this file is in <workflow dir>/.drona/
WORKFLOW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_cli = argparse.ArgumentParser(description="Train with the settings in config.yaml")
_cli.add_argument("--config", default=os.path.join(WORKFLOW_DIR, "config.yaml"),
                  help="settings file (default: config.yaml next to train.py)")
_cli.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                  help="override one setting for this run; repeat as needed")
_args, _ = _cli.parse_known_args()

# All settings, read once: CFG["training"]["max_epochs"], CFG["data"]["batch_size"], ...
CFG = load_config(_args.config, _args.set)


# ── Hardware ─────────────────────────────────────────────────────────────────

def devices_per_node():
    """One training process per task: the job script starts train.py with srun, one task per GPU."""
    return int(os.environ.get("SLURM_NTASKS_PER_NODE", "1"))


def num_nodes():
    return int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))


def apply_matmul_precision(training):
    """Enable TF32 Tensor Core matmuls when config.yaml asks for it.

    Must run before Trainer() so Lightning does not warn on H100/A30.
    """
    level = str((training or {}).get("matmul_precision") or "").strip().lower()
    if level in ("medium", "high"):
        torch.set_float32_matmul_precision(level)


def check_gpu(accelerator):
    """Stop with a clear message when GPU training is asked for but no GPU is visible."""
    if accelerator != "gpu" or torch.cuda.is_available():
        return
    slurm_gpus = os.environ.get("SLURM_JOB_GPUS", "unset")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    raise RuntimeError(
        "GPU training was requested but CUDA is not available on this node. "
        "Verify your Slurm script allocates GPUs (e.g., --gres=gpu:1), or set "
        "training.accelerator: \"cpu\" in config.yaml "
        f"(SLURM_JOB_GPUS={slurm_gpus}, CUDA_VISIBLE_DEVICES={cuda_visible})."
    )


# ── After training ───────────────────────────────────────────────────────────

def has_test_phase(model, datamodule):
    """Run trainer.test only when there is test data and the model defines test_step."""
    if getattr(datamodule, "test_ds", None) is None:
        return False
    return type(model).test_step is not L.LightningModule.test_step


def save_pt_weights(trainer):
    """Save the best and last checkpoints also as plain PyTorch weights (.pt, a state_dict)."""
    checkpoint_callback = trainer.checkpoint_callback
    if not checkpoint_callback or not getattr(checkpoint_callback, "best_model_path", None):
        return
    ckpt_paths = {checkpoint_callback.best_model_path}
    if getattr(checkpoint_callback, "last_model_path", None):
        ckpt_paths.add(checkpoint_callback.last_model_path)
    for ckpt_path in filter(None, ckpt_paths):
        if not os.path.exists(ckpt_path):
            continue
        pt_path = os.path.splitext(ckpt_path)[0] + ".pt"
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            torch.save(ckpt.get("state_dict", ckpt), pt_path)
            print(f"Saved PyTorch weights side-by-side at: {pt_path}")
        except Exception as e:
            print(f"Could not save side-by-side .pt weights from {ckpt_path}: {e}")


def export_onnx(model, datamodule, onnx_path):
    """Export the trained model to ONNX, using one training batch as the example input."""
    try:
        import onnx  # noqa: F401  (to_onnx needs it)
        import inspect
        model.eval()

        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))

        if "edge_index" in inspect.signature(model.forward).parameters:
            x = getattr(batch, "x", None)
            edge_index = getattr(batch, "edge_index", None)
            if x is not None and edge_index is not None:
                input_sample = (x, edge_index)
            else:
                in_channels = getattr(model.hparams, "in_channels", 64)
                input_sample = (torch.randn(10, in_channels), torch.zeros((2, 20), dtype=torch.long))
        else:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            input_sample = x[:1] if isinstance(x, torch.Tensor) else x

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        def _move(t):
            if isinstance(t, torch.Tensor) and t.dtype.is_floating_point:
                return t.to(device=device, dtype=dtype)
            return t.to(device=device) if isinstance(t, torch.Tensor) else t

        input_sample = tuple(_move(t) for t in input_sample) if isinstance(input_sample, tuple) else _move(input_sample)

        os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
        model.to_onnx(onnx_path, input_sample, export_params=True)
        print(f"Successfully exported ONNX model at: {onnx_path}")
    except ImportError:
        print("ERROR: ONNX package not available. Please ensure 'module load GCC/12.3.0 ONNX/1.15.0' is loaded.")
    except Exception as e:
        print(f"Could not export model to ONNX: {e}")
