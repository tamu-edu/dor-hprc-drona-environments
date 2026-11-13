import json
import math
import os
import re
from pathlib import Path

from generators import (
    _gen_computer_vision_script,
    _gen_sequential_script,
    _gen_generative_script,
    _gen_custom_script,
    _gen_nlp_script,
    _gen_tabular_script,
    _gen_audio_script,
    _gen_gnn_script,
)



# ── Utility helpers ────────────────────────────────────────────────────────────

def _normalize_select_value(value):
    """Extract .value from dynamicSelect JSON; pass through plain strings."""
    if value is None:
        return ""
    if isinstance(value, str) and value.startswith("$"):
        return ""
    if isinstance(value, dict):
        return str(value.get("value", "")).strip()
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return str(parsed.get("value", value)).strip()
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return str(value).strip()


def _wants_gpu(gpu, slurm_box):
    if not _checkbox_on(slurm_box):
        return False
    gpu = _normalize_select_value(gpu)
    return gpu not in ("", "none")


def _is_drona_var(val):
    if not val:
        return False
    return str(val).strip().startswith("$")


def _resolve_val(val, default):
    if not val or _is_drona_var(val):
        return default
    return str(val).strip()


def _resolve_precision(precision):
    """Map the form's precision choice to Trainer precision and a TF32 matmul flag.

    Returns (trainer_precision, enable_tf32).
    - tf32 → Trainer precision 32 + torch.set_float32_matmul_precision('high')
    - bf16-mixed / 16-mixed → Lightning mixed precision + TF32 for remaining float32 matmuls
    - 32 → full FP32, no TF32 override
    """
    raw = _normalize_select_value(precision)
    if not raw or _is_drona_var(raw):
        raw = "tf32"
    raw = raw.strip().lower()

    aliases = {
        "tf32": ("32", True),
        "32-tf32": ("32", True),
        "bf16": ("bf16-mixed", True),
        "bf16-mixed": ("bf16-mixed", True),
        "16": ("16-mixed", True),
        "16-mixed": ("16-mixed", True),
        "32": ("32", False),
        "fp32": ("32", False),
    }
    if raw in aliases:
        return aliases[raw]
    if "bf16" in raw:
        return ("bf16-mixed", True)
    if "16-mixed" in raw or raw == "fp16":
        return ("16-mixed", True)
    if "tf32" in raw:
        return ("32", True)
    return (raw, False)


def _inject_tf32_matmul_precision(train_script, enable_tf32):
    """Insert torch.set_float32_matmul_precision('high') before Trainer is built.

    Used when a run falls back to a single generated train.py. The normal path
    sets the same call from config.yaml in files/train.py.
    """
    if not train_script or not enable_tf32:
        return train_script
    if re.search(r"(?m)^\s*torch\.set_float32_matmul_precision\s*\(", train_script):
        return train_script

    block = (
        "# Enable TF32 Tensor Core matmuls on Ampere/Hopper GPUs (e.g. H100, A30)\n"
        'torch.set_float32_matmul_precision("high")\n'
    )
    match = re.search(r"(?m)^(import torch)\s*$", train_script)
    if match:
        idx = match.end()
        return train_script[:idx] + "\n" + block + train_script[idx:]
    return train_script


def _add_matmul_precision(config_yaml, enable_tf32):
    """Record TF32 in config.yaml so the shared train.py enables Tensor Cores."""
    if not config_yaml or not enable_tf32:
        return config_yaml
    if re.search(r"(?m)^  matmul_precision:", config_yaml):
        return config_yaml
    block = (
        "  # TF32 Tensor Core matmuls on Ampere/Hopper GPUs (e.g. H100, A30)\n"
        '  matmul_precision: "high"\n'
    )
    updated, count = re.subn(r"(?m)^(  precision:.*\n)", r"\1" + block, config_yaml, count=1)
    return updated if count else config_yaml


def _resolve_int(val, default):
    if not val or _is_drona_var(val):
        return default
    try:
        return int(str(val).strip())
    except ValueError:
        return default


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
    if not text or text.startswith("$"):
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
    if value is None:
        return 42, None
    text = str(value).strip()
    if not text or text.startswith("$"):
        return 42, None
    try:
        seed = int(text)
    except (TypeError, ValueError):
        return None, "Random seed must be an integer."
    if seed < 0:
        return None, "Random seed must be zero or greater."
    return seed, None


def _get_env_dir():
    p = Path(__file__).resolve().parent
    if p.name == "__pycache__":
        p = p.parent
    return p


# ── Cluster / module setup ─────────────────────────────────────────────────────

# Module stacks. Lightning 2.5.6 (CUDA 12.6) is the default; torchvision and ONNX
# only exist for GCC/12.3.0 toolchains, so those features switch to the 2.2.1 stack
# (loading both would trigger an Lmod toolchain swap that deactivates the CUDA-12.6.0 packages).
STACK_DEFAULT = "module load GCC/13.3.0 OpenMPI/5.0.3 Lightning/2.5.6-CUDA-12.6.0"
STACK_LEGACY = (
    "module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch-Lightning/2.2.1-CUDA-12.1.1\n"
    "module load CUDA/12.1.1 2>/dev/null || true"
)
STACK_GNN = (
    "module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch-Geometric/2.1.0-PyTorch-1.12.0-CUDA-11.7.0\n"
    "module load PyTorch-Lightning/1.7.7-CUDA-11.7.0 2>/dev/null || true\n"
    "module load CUDA/11.7.0 2>/dev/null || true"
)
TORCHVISION_MODULE = "torchvision/0.16.0-CUDA-12.1.1"
ONNX_MODULE_CMD = "module load GCC/12.3.0 ONNX/1.15.0"

# Accelerators the GNN stack (CUDA 11.7) cannot run on
GNN_UNSUPPORTED_GPUS = ("h100",)


def _select_stack(model_category, uses_torchvision, export_onnx):
    """Return (base modules, extra module commands) for a model category and feature set."""
    if model_category == "gnn":
        return STACK_GNN, ""
    extra = ""
    base = STACK_DEFAULT
    if uses_torchvision:
        base = STACK_LEGACY
        extra += f"\n# Load torchvision module\nmodule load {TORCHVISION_MODULE} 2>/dev/null || true\n"
    if export_onnx:
        base = STACK_LEGACY
        extra += f"\n# Load ONNX cluster module\n{ONNX_MODULE_CMD}\n"
    return base, extra


CV_DATASET_MODULES = {
    "ImageNet": "Datasets/ImageNet-PyTorch/2012",
    "COCO": "Datasets/COCO/2017",
    "CC3M": "Datasets/CC3M/1.1",
    "CAMUS": "Datasets/CAMUS/2019",
}

SEQ_DATASET_MODULES = {
    "fastText": "Datasets/fastText/2015",
    "AISHELL": "Datasets/AISHELL/2017",
}



GEN_DATASET_MODULES = {
    "llava-onevision": "Datasets/llava-onevision/2024",
}


def setup_pytorch_modules(
    model_category="computer_vision",
    dataset_type="builtin",
    builtin_dataset="ImageNet",
    seq_dataset="mackey_glass",
    graph_dataset="cora",
    gen_dataset="llava-onevision",
    nlp_dataset_type="builtin",
    nlp_builtin_dataset="fastText",
    tab_dataset_type="builtin",
    tab_builtin_dataset="Synthetic Tabular",
    audio_dataset_type="builtin",
    audio_builtin_dataset="AISHELL",
    gpu="",
    graph_dataset_type="builtin",
    export_onnx="No",
    seq_dataset_type="builtin",
    gen_dataset_type="builtin",
):
    """Return module load commands for the given model category and dataset."""
    ds_t = (dataset_type or "builtin").strip()
    dataset_cmd = ""
    
    if model_category == "nlp":
        ds_t = (nlp_dataset_type or "builtin").strip()
    elif model_category == "tabular":
        ds_t = (tab_dataset_type or "builtin").strip()
    elif model_category == "audio":
        ds_t = (audio_dataset_type or "builtin").strip()
    elif model_category == "gnn":
        ds_t = (graph_dataset_type or "builtin").strip()
    elif model_category == "sequential":
        ds_t = (seq_dataset_type or "builtin").strip()
    elif model_category == "generative":
        ds_t = (gen_dataset_type or "builtin").strip()

    if ds_t == "builtin":
        if model_category == "computer_vision":
            if builtin_dataset in CV_DATASET_MODULES:
                dataset_cmd = f"\nmodule load {CV_DATASET_MODULES[builtin_dataset]} 2>/dev/null || true"
        elif model_category == "sequential":
            if seq_dataset in SEQ_DATASET_MODULES:
                dataset_cmd = f"\nmodule load {SEQ_DATASET_MODULES[seq_dataset]} 2>/dev/null || true"
        elif model_category == "generative":
            if gen_dataset in GEN_DATASET_MODULES:
                dataset_cmd = f"\nmodule load {GEN_DATASET_MODULES[gen_dataset]} 2>/dev/null || true"
        elif model_category == "nlp":
            if nlp_builtin_dataset in SEQ_DATASET_MODULES:
                dataset_cmd = f"\nmodule load {SEQ_DATASET_MODULES[nlp_builtin_dataset]} 2>/dev/null || true"
        elif model_category == "audio":
            if audio_builtin_dataset in SEQ_DATASET_MODULES:
                dataset_cmd = f"\nmodule load {SEQ_DATASET_MODULES[audio_builtin_dataset]} 2>/dev/null || true"
        elif model_category == "gnn":
            if graph_dataset == "QM9":
                dataset_cmd = "\nmodule load Datasets/QM9/2012 2>/dev/null || true"

    uses_torchvision = model_category == "computer_vision" or (
        model_category == "generative"
        and (ds_t == "custom" or gen_dataset == "llava-onevision")
    )
    base, extra = _select_stack(model_category, uses_torchvision, _checkbox_on(export_onnx))

    return (
        "# Load cluster PyTorch Lightning stack and datasets\n"
        f"{base}{extra}{dataset_cmd}"
    )


def setup_pytorch_modules_if_run(
    mode,
    model_category="computer_vision",
    dataset_type="builtin",
    builtin_dataset="ImageNet",
    seq_dataset="mackey_glass",
    graph_dataset="cora",
    gen_dataset="llava-onevision",
    nlp_dataset_type="builtin",
    nlp_builtin_dataset="fastText",
    tab_dataset_type="builtin",
    tab_builtin_dataset="Synthetic Tabular",
    audio_dataset_type="builtin",
    audio_builtin_dataset="AISHELL",
    gpu="",
    graph_dataset_type="builtin",
    export_onnx="No",
    seq_dataset_type="builtin",
    gen_dataset_type="builtin",
):
    if mode == "manage":
        return "# manage mode — no training job"
    return setup_pytorch_modules(
        model_category, dataset_type, builtin_dataset, seq_dataset, graph_dataset, gen_dataset,
        nlp_dataset_type, nlp_builtin_dataset, tab_dataset_type, tab_builtin_dataset, audio_dataset_type, audio_builtin_dataset,
        gpu, graph_dataset_type, export_onnx, seq_dataset_type, gen_dataset_type
    )


def retrieve_driver_contents(mode, cancel_jobs=""):
    base = _get_env_dir()
    if mode == "manage":
        file_path = base / "drivers" / "driver-manage.sh"
    else:
        # create, or the mode not chosen yet
        file_path = base / "drivers" / "driver-create.sh"
        if mode == "create":
            drona_add_additional_file("jobscripts/lightning-job.slurm", "Job Script")
            drona_add_additional_file("jobscripts/env.sh", "env.sh")
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def retrieve_pt_slurm_params(mode, model_category, nodes, num_workers, mem, gpu, numgpu, walltime, account, extra):
    """Shared retrieve_slurm_params with tasks and cpus derived from the training setup:
    one task per GPU on each node (one per node without GPUs), DataLoader workers + 1 CPUs per task.
    Also applies the GNN accelerator restriction (PyG stack is CUDA 11.7)."""
    from drona_runtime_utils import retrieve_slurm_params
    from drona_utils import drona_add_mapping

    if mode == "manage":
        return ""
    if model_category == "gnn" and _normalize_select_value(gpu) in GNN_UNSUPPORTED_GPUS:
        drona_add_message(
            "Graph Neural Networks use a CUDA 11.7 stack that does not support the selected accelerator. Using A30 instead.",
            "warning",
        )
        gpu = "a30"

    nodenum = max(_resolve_int(nodes, 1), 1)
    tasks_per_node = 1
    if _normalize_select_value(gpu) not in ("", "none"):
        tasks_per_node = max(_resolve_int(numgpu, 1), 1)
    cpus = max(_resolve_int(num_workers, 0), 0) + 1

    # The cluster module may adjust the node count and cap the GPUs per node; derive
    # --ntasks/--ntasks-per-node from its final values (it adds TASKS, NODES, then PARTITION)
    final = {}

    def mapping_filter(key, value):
        if key in ("TASKS", "NODES"):
            final[key] = int(value)
            if key == "NODES" and "TASKS" in final:
                final["TASKS_PER_NODE"] = math.ceil(final["TASKS"] / final["NODES"])
                drona_add_mapping("TASKS_PER_NODE", str(final["TASKS_PER_NODE"]))
        elif key == "PARTITION" and tasks_per_node > 1 and "NODES" in final:
            gres = re.search(r"--gres=gpu:(?:[^:\s]+:)?(\d+)", value or "")
            if gres and int(gres.group(1)) < final.get("TASKS_PER_NODE", 0):
                capped = int(gres.group(1))
                drona_add_mapping("TASKS", str(capped * final["NODES"]))
                drona_add_mapping("TASKS_PER_NODE", str(capped))
        return value

    return retrieve_slurm_params(str(nodenum), str(nodenum * tasks_per_node), str(cpus), mem, gpu, numgpu,
                                 walltime, account, extra, mapping_filter=mapping_filter)


# ── Drona internal helpers ─────────────────────────────────────────────────────

def _normalize_job_dir(job_dir):
    job_dir_str = str(job_dir) if job_dir is not None else ""
    match = re.search(r">(/[^<]+)<", job_dir_str)
    return (match.group(1) if match else job_dir_str.strip())


def _normalize_location(location):
    loc = _normalize_job_dir(location)
    if not loc or loc.startswith("$"):
        return ""
    return loc


def _write_staged_file(env_dir, job_location, filename, content, show_in_preview=True, preview_order=None):
    """Add a generated file to the preview. Nothing is written now: Drona writes it into the
    workflow directory at submit, so a preview never overwrites files the user edited there.
    filename may be a relative path (e.g. ".drona/helper.py"); hidden files use preview_order=-1."""
    if preview_order is None:
        preview_order = 0 if show_in_preview else -1
    drona_add_additional_file(content=content, name=filename, preview_name=filename,
                              preview_order=preview_order)


def retrieve_workflow_actions(mode, delete_workflow, delete_files, allworkflows, job_dir=""):
    """Manage-mode script: delete the selected workflow's record from the database when
    "Remove from history" is checked (and "Also delete files" optionally removes the directory too). The working directory is left alone. The record of this
    manage run itself ($DRONA_WF_ID) is always cleaned up."""
    import json, re, shlex
    from views.utils import get_runtime_dir
    from drona_utils.core import drona_add_warning, drona_add_error

    if not mode or mode != "manage":
        return "echo 'No action taken.'"

    db_retriever = os.path.join(get_runtime_dir(), "db_access", "drona_db_retriever.py")

    def step(title, target):
        cmd = f"python3 {db_retriever} --delete -i {target}"
        return (
            f'echo "==> {title}"\n'
            f'echo "+ {cmd}"\n'
            f"{cmd} && echo 'Done.' || echo 'FAILED: exit code '$?'.'"
        )

    # Left out for a run on an existing workflow (no throwaway record; DRONA_WF_ID is the
    # workflow itself). The DRONA_RUN_DIR test guards scripts generated the old way.
    import drona_utils.core as _du
    on_target = getattr(_du, "drona_get_run_target", lambda: None)()
    cleanup = "" if on_target else (
        'if [ -z "${DRONA_RUN_DIR:-}" ]; then\n'
        + step("Cleaning up the record of this manage run ($DRONA_WF_ID)", "$DRONA_WF_ID")
        + "\nfi"
    )

    # Checkbox values arrive as "" when unchecked, or an unresolved "$name" when never rendered
    def checked(v):
        return bool(v) and not str(v).startswith("$")

    option = "workflow_files" if checked(delete_workflow) and checked(delete_files) else "workflow" if checked(delete_workflow) else ""
    wf_id = (allworkflows or "").strip()
    if option not in ("workflow", "workflow_files") or not wf_id or wf_id.startswith("$"):
        return cleanup

    wf = shlex.quote(wf_id)
    delete_wf = step(f'Deleting workflow {wf_id} from the Drona database', wf)

    # Optional: also remove the working directory from disk
    if option != "workflow_files":
        return f"{delete_wf}\necho 'Working directory kept on disk.'\n{cleanup}"

    match = re.search(r'>(/[^<]+)<', job_dir or "")
    path = (match.group(1) if match else (job_dir or "")).strip()
    real = os.path.realpath(path) if path.startswith("/") else ""
    home = os.path.realpath(os.path.expanduser("~"))
    # refuse anything that is not a clearly dedicated directory (root, $HOME or its parents, top-level dirs)
    if not real or len(Path(real).parts) < 4 or real == home or home.startswith(real.rstrip("/") + "/"):
        drona_add_error(f"Refusing to delete the working directory: unsafe or missing path '{path}'. Only the database record will be deleted.")
        return f"{delete_wf}\necho 'Working directory NOT deleted: unsafe or missing path.'\n{cleanup}"

    drona_add_warning(f"The working directory {real} and ALL of its contents will be permanently deleted. This cannot be undone.")
    q = shlex.quote(real)
    rm = (
        f'echo "==> Deleting working directory {real} from disk"\n'
        f'echo "+ rm -rf {q}"\n'
        f"cd / && rm -rf {q} && echo 'Done.' || echo 'FAILED: exit code '$?'.'"
    )
    return f"{delete_wf}\n{rm}\n{cleanup}"

# ── Shared template blocks embedded in generated scripts ──────────────────────

_DOWNLOAD_ONLY_BLOCK = ""


_DATA_HELPERS_BLOCK = ""


_LIGHTNING_IMPORT_BLOCK = '''
# Configure PyTorch Lightning / Lightning package imports (compatibility wrapper)
try:
    import pytorch_lightning as L
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
except ImportError:
    import lightning as L
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
'''


# Script generators are imported from the generators package.



# ── Main dispatcher ────────────────────────────────────────────────────────────

def externalize_settings(train_script, header=""):
    """generators/config_file.py, loaded by path: the app keeps one `generators` package in
    sys.modules for all environments, so an import could pick up another env's copy."""
    import importlib.util
    path = Path(__file__).resolve().parent / "generators" / "config_file.py"
    spec = importlib.util.spec_from_file_location("_drona_pl_config_file", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.externalize_settings(train_script, header)


_CONFIG_HEADER = """\
# Settings for this training run, generated by Drona from the form.
# Edit the values here and resubmit; train.py reads this file at start.
# One-off change without editing:  python3 train.py --set training.max_epochs=20
# What is not here (dataset choice, network layout) is in data.py and model.py."""

_CONFIG_OUTPUTS = """\
  # Keep the best (lowest val_loss) and last checkpoint, plus a .pt copy of the weights
  checkpoint: {checkpoint}
  # Export the trained model to ONNX (log_dir/final_model.onnx)
  export_onnx: {export_onnx}
"""


def split_into_files(train_script):
    """generators/split_files.py, loaded by path (same reason as externalize_settings)."""
    import importlib.util
    path = Path(__file__).resolve().parent / "generators" / "split_files.py"
    spec = importlib.util.spec_from_file_location("_drona_pl_split_files", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.split_script(train_script)


def _static_file(name):
    """A file from files/ in this environment, e.g. the shared train.py."""
    with open(Path(__file__).resolve().parent / "files" / name, encoding="utf-8") as f:
        return f.read()


def generate_lightning_script_if_run(
    mode,
    modelCategory="computer_vision",
    name="lightning_run",
    datasetType="builtin",
    builtinDataset="MNIST",
    customDataPath="",
    epochs="10",
    batchSize="32",
    learningRate="1e-3",
    numWorkers="0",
    seed="42",
    gpu="",
    logDir="./lightning_logs",
    logEveryNSteps="50",
    enableTensorBoard="Yes",
    checkpointEnable="Yes",
    hfModelName="",
    hfDatasetName="",
    textDatasetType="builtin",
    textColumn="",
    labelColumn="",
    customTextPath="",
    maxSeqLen="128",
    numLabels="2",
    seqDatasetType="builtin",
    seqBuiltinDataset="mackey_glass",
    tsDataPath="",
    seqLen="50",
    predLen="1",
    targetColumn="target",
    hiddenSize="128",
    numLstmLayers="2",
    seqTaskType="regression",
    graphDatasetType="builtin",
    graphBuiltinDataset="cora",
    graphDataPath="",
    gnnHiddenDim="64",
    gnnNumLayers="2",
    genDatasetType="builtin",
    genBuiltinDataset="MNIST",
    genCustomPath="",
    latentDim="128",
    customDatasetPath="",
    job_location="",
    slurmBox="Yes",
    precision="tf32",
    cvModelArch="cnn",
    seqModelType="lstm",
    gnnLayerType="gcn",
    genModelType="vae",
    cvDatasetFormat="tensors",
    genDatasetFormat="tensors",
    exportOnnx="No",
    nlpDatasetType="custom",
    nlpBuiltinDataset="fastText",
    nlpCustomPath="",
    nlpModelType="attention",
    nlpMaxSeqLen="128",
    tabDatasetType="custom",
    tabBuiltinDataset="synthetic_tabular",
    tabCustomPath="",
    tabTargetColumn="target",
    tabModelType="mlp",
    tabTaskType="regression",
    audioDatasetType="custom",
    audioBuiltinDataset="AISHELL",
    audioCustomPath="",
    audioModelType="classification",
    audioTransformType="mel_spectrogram",
    cvImageSize="0",
    cvChannels="3",
):
    if mode != "create":
        return ""
    # Tasks whose example dataset is hidden in the form (missing or unreadable cluster copy).
    # An old workflow restored with Recreate can still send "builtin" for them.
    no_example = {"nlp": nlpDatasetType, "audio": audioDatasetType,
                  "generative": genDatasetType, "gnn": graphDatasetType}
    if modelCategory in no_example and str(no_example[modelCategory]).strip() == "builtin":
        from drona_utils.core import drona_add_error
        drona_add_error("This task has no example dataset yet. Under DATA, choose 'My own data' "
                        "and point to your dataset.")
        return ""
    return generate_lightning_script(
        modelCategory,
        name, datasetType, builtinDataset, customDataPath,
        epochs, batchSize, learningRate, numWorkers, seed, gpu,
        logDir, logEveryNSteps,
        enableTensorBoard, checkpointEnable,
        hfModelName, hfDatasetName, textDatasetType, textColumn, labelColumn, customTextPath, maxSeqLen, numLabels,
        seqDatasetType, seqBuiltinDataset, tsDataPath, seqLen, predLen, targetColumn, hiddenSize, numLstmLayers, seqTaskType,
        graphDatasetType, graphBuiltinDataset, graphDataPath, gnnHiddenDim, gnnNumLayers,
        genDatasetType, genBuiltinDataset, genCustomPath, latentDim,
        customDatasetPath,
        job_location,
        slurmBox,
        precision,
        cvModelArch,
        seqModelType,
        gnnLayerType,
        genModelType,
        cvDatasetFormat,
        genDatasetFormat,
        exportOnnx,
        nlpDatasetType,
        nlpBuiltinDataset,
        nlpCustomPath,
        nlpModelType,
        nlpMaxSeqLen,
        tabDatasetType,
        tabBuiltinDataset,
        tabCustomPath,
        tabTargetColumn,
        tabModelType,
        tabTaskType,
        audioDatasetType,
        audioBuiltinDataset,
        audioCustomPath,
        audioModelType,
        audioTransformType,
        cvImageSize,
        cvChannels,
    )


def generate_lightning_script(
    modelCategory="computer_vision",
    name="lightning_run",
    datasetType="builtin",
    builtinDataset="MNIST",
    customDataPath="",
    epochs="10",
    batchSize="32",
    learningRate="1e-3",
    numWorkers="0",
    seed="42",
    gpu="",
    logDir="./lightning_logs",
    logEveryNSteps="50",
    enableTensorBoard="Yes",
    checkpointEnable="Yes",
    hfModelName="",
    hfDatasetName="",
    textDatasetType="builtin",
    textColumn="",
    labelColumn="",
    customTextPath="",
    maxSeqLen="128",
    numLabels="2",
    seqDatasetType="builtin",
    seqBuiltinDataset="mackey_glass",
    tsDataPath="",
    seqLen="50",
    predLen="1",
    targetColumn="target",
    hiddenSize="128",
    numLstmLayers="2",
    seqTaskType="regression",
    graphDatasetType="builtin",
    graphBuiltinDataset="cora",
    graphDataPath="",
    gnnHiddenDim="64",
    gnnNumLayers="2",
    genDatasetType="builtin",
    genBuiltinDataset="MNIST",
    genCustomPath="",
    latentDim="128",
    customDatasetPath="",
    job_location="",
    slurmBox="Yes",
    precision="tf32",
    cvModelArch="cnn",
    seqModelType="lstm",
    gnnLayerType="gcn",
    genModelType="vae",
    cvDatasetFormat="tensors",
    genDatasetFormat="tensors",
    exportOnnx="No",
    nlpDatasetType="custom",
    nlpBuiltinDataset="fastText",
    nlpCustomPath="",
    nlpModelType="attention",
    nlpMaxSeqLen="128",
    tabDatasetType="custom",
    tabBuiltinDataset="synthetic_tabular",
    tabCustomPath="",
    tabTargetColumn="target",
    tabModelType="mlp",
    tabTaskType="regression",
    audioDatasetType="custom",
    audioBuiltinDataset="AISHELL",
    audioCustomPath="",
    audioModelType="classification",
    audioTransformType="mel_spectrogram",
    cvImageSize="0",
    cvChannels="3",
):
    # ── Parse & validate shared params ────────────────────────────────────────
    category = (modelCategory or "computer_vision").strip()
    exp_name = sanitize_job_name(name) or "lightning_run"

    lr, lr_err = _parse_learning_rate(learningRate)
    if lr_err:
        drona_add_message(lr_err, "error")
        return ""

    seed_val, seed_err = _parse_seed(seed)
    if seed_err:
        drona_add_message(seed_err, "error")
        return ""

    ep  = _resolve_int(epochs, 10)
    bs  = _resolve_int(batchSize, 32)
    nw  = _resolve_int(numWorkers, 0)
    log_n = _resolve_int(logEveryNSteps, 50)
    log_dir = _resolve_val(logDir, "./lightning_logs")

    acc = "gpu" if _wants_gpu(gpu, slurmBox) else "cpu"
    # One process per task: the job script launches train.py with srun and
    # retrieve_pt_slurm_params sets --ntasks-per-node to the GPUs per node
    dev = 'int(os.environ.get("SLURM_NTASKS_PER_NODE", "1"))'
    prec, enable_tf32 = _resolve_precision(precision)

    tb_on   = True
    ckpt_on = _checkbox_on(checkpointEnable)

    logger_lines = []
    if tb_on:
        logger_lines.append(
            f'    TensorBoardLogger(save_dir=LOG_DIR, name="{_py_str(exp_name)}"),'
        )
    if not logger_lines:
        logger_lines.append("    False,")

    callback_lines = []
    if ckpt_on:
        callback_lines.append(
            '        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=True),'
        )
    callback_block = (
        "callbacks = [\n" + "\n".join(callback_lines) + "\n    ]"
        if callback_lines
        else "callbacks = None"
    )

    env_dir = _get_env_dir()

    # ── Dispatch to category generator ────────────────────────────────────────
    train_script = prefetch_script = None

    if category == "computer_vision":
        ds_type = datasetType or "builtin"
        builtin = builtinDataset or "MNIST"
        custom_path = customDataPath or ""
        if ds_type == "custom" and not custom_path:
            drona_add_message("Custom dataset path is required.", "error")
            return ""
        train_script, prefetch_script = _gen_computer_vision_script(
            exp_name, ds_type, builtin, custom_path,
            ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
            logger_lines, callback_block,
            cv_model_arch=cvModelArch,
            cv_dataset_format=cvDatasetFormat,
            cv_image_size=cvImageSize,
            cv_channels=cvChannels,
        )

    elif category == "sequential":
        seq_ds_type = (seqDatasetType or "builtin").strip()
        seq_blt     = (seqBuiltinDataset or "mackey_glass").strip()
        ts_path     = (tsDataPath or "").strip()
        tgt_col     = (targetColumn or "target").strip()
        task_type   = (seqTaskType or "regression").strip()
        s_len       = int(seqLen)      if seqLen      else 50
        p_len       = int(predLen)     if predLen     else 1
        h_size      = int(hiddenSize)  if hiddenSize  else 128
        n_layers    = int(numLstmLayers) if numLstmLayers else 2
        
        train_script, prefetch_script = _gen_sequential_script(
            exp_name, ts_path, tgt_col, task_type,
            s_len, p_len, h_size, n_layers,
            ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
            logger_lines, callback_block,
            seq_dataset_type=seq_ds_type,
            seq_builtin_dataset=seq_blt,
            seq_model_type=seqModelType,
        )

    elif category == "gnn":
        graph_ds_type = (graphDatasetType or "builtin").strip()
        if _is_drona_var(graph_ds_type) or not graph_ds_type:
            graph_ds_type = "builtin"
        
        graph_blt = (graphBuiltinDataset or "QM9").strip()
        if _is_drona_var(graph_blt) or not graph_blt:
            graph_blt = "QM9"
            
        graph_path = (graphDataPath or "").strip()
        if _is_drona_var(graph_path):
            graph_path = ""
            
        gnn_hidden = _resolve_int(gnnHiddenDim, 64)
        gnn_layers = _resolve_int(gnnNumLayers, 2)
        
        gnn_layer = (gnnLayerType or "gcn").strip().lower()
        if _is_drona_var(gnn_layer) or not gnn_layer:
            gnn_layer = "gcn"

        if graph_ds_type == "custom" and not graph_path:
            drona_add_message("Custom Graph dataset path is required.", "error")
            return ""

        train_script, prefetch_script = _gen_gnn_script(
            exp_name, graph_ds_type, graph_blt, graph_path,
            ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
            logger_lines, callback_block,
            gnn_hidden=gnn_hidden,
            gnn_layers=gnn_layers,
            gnn_layer=gnn_layer,
        )

    elif category == "generative":
        gen_ds_type = (genDatasetType or "builtin").strip()
        gen_blt     = (genBuiltinDataset or "MNIST").strip()
        gen_type    = gen_blt if gen_ds_type == "builtin" else "custom"
        gen_path    = (genCustomPath or "").strip()
        lat_dim     = int(latentDim) if latentDim else 128
        
        train_script, prefetch_script = _gen_generative_script(
            exp_name, gen_type, gen_path, lat_dim,
            ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
            logger_lines, callback_block,
            gen_model_type=genModelType,
            gen_dataset_format=genDatasetFormat,
        )

    elif category == "nlp":
        nlp_ds_type = (nlpDatasetType or "custom").strip()
        nlp_blt     = (nlpBuiltinDataset or "fastText").strip()
        nlp_path    = (nlpCustomPath or "").strip()
        nlp_max_len = int(nlpMaxSeqLen) if nlpMaxSeqLen else 128
        
        train_script, prefetch_script = _gen_nlp_script(
            exp_name, nlp_ds_type, nlp_blt, nlp_path, nlp_max_len,
            ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
            logger_lines, callback_block,
            nlp_model_type=nlpModelType,
        )

    elif category == "tabular":
        tab_ds_type = (tabDatasetType or "custom").strip()
        tab_blt     = (tabBuiltinDataset or "synthetic_tabular").strip()
        tab_path    = (tabCustomPath or "").strip()
        tab_tgt     = (tabTargetColumn or "target").strip()
        tab_task    = (tabTaskType or "regression").strip()
        
        train_script, prefetch_script = _gen_tabular_script(
            exp_name, tab_ds_type, tab_blt, tab_path, tab_tgt, tab_task,
            ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
            logger_lines, callback_block,
            tab_model_type=tabModelType,
        )

    elif category == "audio":
        audio_ds_type = (audioDatasetType or "custom").strip()
        audio_blt     = (audioBuiltinDataset or "AISHELL").strip()
        audio_path    = (audioCustomPath or "").strip()
        
        train_script, prefetch_script = _gen_audio_script(
            exp_name, audio_ds_type, audio_blt, audio_path,
            ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
            logger_lines, callback_block,
            audio_model_type=audioModelType,
            audio_transform_type=audioTransformType,
        )

    elif category == "custom":
        train_script, prefetch_script = _gen_custom_script(
            exp_name, ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
            logger_lines, callback_block,
            custom_dataset_path=customDatasetPath,
        )

    else:
        drona_add_message(f"Unknown model category: {category}", "error")
        return ""

    if train_script is None:
        return ""  # error already added by sub-generator

    export_onnx = _checkbox_on(exportOnnx) and category != "gnn"
    # The Custom task has no validation step, so there is no val_loss to pick checkpoints by
    checkpoint = ckpt_on and category != "custom"

    # ── Files: config.yaml + train.py (the same for every task) + data.py + model.py ──
    # Nothing is written now: Drona writes them into the workflow directory at submit.
    train_script, config_yaml = externalize_settings(train_script, header=_CONFIG_HEADER)
    try:
        data_py, model_py = split_into_files(train_script)
    except Exception as e:
        # Unexpected script shape: fall back to the single train.py of earlier versions
        drona_add_message(f"PyTorch-Lightning: could not split the script into data.py and model.py "
                          f"({e}); writing a single train.py instead.", "warning")
        if export_onnx:
            train_script = _inject_onnx_export(train_script)
        train_script = _inject_tf32_matmul_precision(train_script, enable_tf32)
        if config_yaml:
            _write_staged_file(env_dir, job_location, "config.yaml", _add_matmul_precision(config_yaml, enable_tf32))
        _write_staged_file(env_dir, job_location, "train.py", train_script)
        return ""

    config_yaml = _add_matmul_precision(
        (config_yaml or "").rstrip("\n") + "\n" + _CONFIG_OUTPUTS.format(
            checkpoint="true" if checkpoint else "false", export_onnx="true" if export_onnx else "false"),
        enable_tf32,
    )
    readme = _build_readme(category, exp_name, _dataset_label(category, locals()), ep, bs, lr, prec,
                           log_dir, checkpoint, export_onnx)
    _write_staged_file(env_dir, job_location, "README.md", readme)
    _write_staged_file(env_dir, job_location, "config.yaml", config_yaml)
    _write_staged_file(env_dir, job_location, "train.py", _static_file("train.py"))
    _write_staged_file(env_dir, job_location, "data.py", data_py)
    _write_staged_file(env_dir, job_location, "model.py", model_py)
    _write_staged_file(env_dir, job_location, ".drona/pl_support.py", _static_file("pl_support.py"),
                       show_in_preview=False)
    return ""


_TASK_TITLES = {
    "computer_vision": "Computer vision model",
    "sequential": "Sequential / time-series model",
    "gnn": "Graph neural network",
    "generative": "Generative model",
    "nlp": "Natural language processing model",
    "tabular": "Tabular data model",
    "audio": "Audio model",
    "custom": "Bare-bones custom model",
}
_EXAMPLE_NAMES = {"mackey_glass": "Mackey-Glass (synthetic)", "synthetic_tabular": "Synthetic tabular (generated)",
                  "llava-onevision": "LLaVA-OneVision"}

# Per task: (dataset type, example dataset name, custom data path) variables in generate_lightning_script
_DATASET_VARS = {
    "computer_vision": ("ds_type", "builtin", "custom_path"),
    "sequential": ("seq_ds_type", "seq_blt", "ts_path"),
    "gnn": ("graph_ds_type", "graph_blt", "graph_path"),
    "generative": ("gen_ds_type", "gen_blt", "gen_path"),
    "nlp": ("nlp_ds_type", "nlp_blt", "nlp_path"),
    "tabular": ("tab_ds_type", "tab_blt", "tab_path"),
    "audio": ("audio_ds_type", "audio_blt", "audio_path"),
}


def _dataset_label(category, values):
    """'the example dataset X' or 'your data in `path`', for the README."""
    if category == "custom":
        path = (values.get("customDatasetPath") or "").strip()
        return f"your data in `{path}`" if path else "the placeholder data in data.py"
    type_var, name_var, path_var = _DATASET_VARS[category]
    if values.get(type_var) == "custom":
        return f"your data in `{values.get(path_var) or ''}`"
    name = values.get(name_var) or ""
    return f"the example dataset {_EXAMPLE_NAMES.get(name, name)}"


def _build_readme(category, name, dataset, epochs, batch_size, lr, precision, log_dir, checkpoint, export_onnx):
    run_dir = f"{log_dir}/{name}/version_N"
    checkpoint_row = (f"| Checkpoints (best and last, plus `.pt` weights) | `{run_dir}/checkpoints/` |\n"
                      if checkpoint else "")
    onnx_row = f"| ONNX model | `{log_dir}/final_model.onnx` |\n" if export_onnx else ""
    return _static_file("README.md").format(
        name=name, title=_TASK_TITLES.get(category, "PyTorch Lightning model"), dataset=dataset,
        max_epochs=epochs, batch_size=batch_size, learning_rate=lr, precision=precision,
        log_dir=log_dir, experiment=name, checkpoint_row=checkpoint_row, onnx_row=onnx_row)


def _inject_onnx_export(train_script):
    """Single-file fallback only: export to ONNX right after trainer.fit."""
    onnx_export_code = '''    trainer.fit(model, datamodule=datamodule)

    # Export final model to ONNX
    try:
        import onnx
        import inspect
        model.eval()
        
        # Setup datamodule and get a sample batch
        datamodule.setup("fit")
        dl = datamodule.train_dataloader()
        batch = next(iter(dl))
        
        sig = inspect.signature(model.forward)
        
        if "edge_index" in sig.parameters:
            x = getattr(batch, 'x', None)
            edge_index = getattr(batch, 'edge_index', None)
            if x is not None and edge_index is not None:
                input_sample = (x, edge_index)
            else:
                in_channels = getattr(model.hparams, 'in_channels', 64)
                input_sample = (torch.randn(10, in_channels), torch.zeros((2, 20), dtype=torch.long))
        else:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
                
            if isinstance(x, torch.Tensor):
                input_sample = x[:1]
            else:
                input_sample = x
        
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        if isinstance(input_sample, tuple):
            input_sample = tuple(
                t.to(device=device, dtype=dtype) if isinstance(t, torch.Tensor) and t.dtype.is_floating_point else t.to(device=device)
                for t in input_sample
            )
        elif isinstance(input_sample, torch.Tensor):
            if input_sample.dtype.is_floating_point:
                input_sample = input_sample.to(device=device, dtype=dtype)
            else:
                input_sample = input_sample.to(device=device)
                
        onnx_path = os.path.join(LOG_DIR, "final_model.onnx")
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        model.to_onnx(onnx_path, input_sample, export_params=True)
        print(f"Successfully exported ONNX model at: {onnx_path}")
    except ImportError:
        print("ERROR: ONNX package not available. Please ensure 'module load GCC/12.3.0 ONNX/1.15.0' is loaded.")
    except Exception as e:
        print(f"Could not export model to ONNX: {e}")'''
    return train_script.replace("    trainer.fit(model, datamodule=datamodule)", onnx_export_code)


def setup_tensorboard_in_job(mode, logDir="", enable="yes"):
    # enable: the MONITORING "TensorBoard" checkbox; any non-empty value except no/false/0 is on
    if mode != "create":
        return ""
    if str(enable or "").strip().lower() in ("", "no", "false", "0"):
        return ""

    # Same value the generated train.py gets as LOG_DIR
    log_dir = _resolve_val(logDir, "./lightning_logs")
    if log_dir.startswith("./"):
        log_dir = log_dir[2:]

    return f"""# TensorBoard log directory: logging.log_dir in config.yaml if set there, else the form's value
TB_LOGDIR="{log_dir}"
if [ -f config.yaml ]; then
    _cfg_logdir=$(grep -E '^\\s+log_dir:' config.yaml | head -n 1 | sed -E 's/^\\s+log_dir:\\s*//; s/\\s+#.*$//; s/^"(.*)"$/\\1/')
    [ -n "$_cfg_logdir" ] && TB_LOGDIR="${{_cfg_logdir#./}}"
fi
export TB_LOGDIR

# Launch TensorBoard in the background on the compute node using srun
if [ -d "$TB_LOGDIR" ] || mkdir -p "$TB_LOGDIR"; then
    echo "Starting TensorBoard background server with srun..." >&2
    
    # Inline port finder logic: find a free port in range 6006-6050, then random high ports
    TB_PORT=""
    for port in $(seq 6006 6050); do
      if ! ss -tuln 2>/dev/null | grep -q ":${{port}} "; then
        TB_PORT=$port
        break
      fi
    done
    if [ -z "$TB_PORT" ]; then
      for i in $(seq 1 1000); do
        rand_val=$(( (RANDOM << 15) | RANDOM ))
        port=$(( 6051 + (rand_val % 59485) ))
        if ! ss -tuln 2>/dev/null | grep -q ":${{port}} "; then
          TB_PORT=$port
          break
        fi
      done
      if [ -z "$TB_PORT" ]; then
        for port in $(seq 6051 65535); do
          if ! ss -tuln 2>/dev/null | grep -q ":${{port}} "; then
            TB_PORT=$port
            break
          fi
        done
      fi
    fi
    if [ -z "$TB_PORT" ]; then
      echo "ERROR: No free port found on node $(hostname)" >&2
      exit 1
    fi

    echo "$TB_PORT" > tb_port.txt
    export TB_PORT
    NODE_NAME=$(hostname)
    
    # Launch tensorboard concurrently within the main job's allocation using module loading inside srun
    (
        srun --ntasks=1 --nodes=1 --cpus-per-task=1 --overlap bash -c '
            source /etc/profile
            if [ -f /sw/lmod/lmod/init/bash ]; then
                source /sw/lmod/lmod/init/bash
            elif [ -f /usr/share/lmod/lmod/init/bash ]; then
                source /usr/share/lmod/lmod/init/bash
            fi
            module load GCC/13.2.0 tensorboard/2.18.0 2>/dev/null || \\
            module load GCC/12.3.0 tensorboard/2.17.0 2>/dev/null || \\
            module load GCC/12.3.0 tensorboard/2.15.1-CUDA-12.1.1 2>/dev/null || \\
            module load tensorboard 2>/dev/null || true

            if command -v tensorboard >/dev/null 2>&1; then
                tensorboard --logdir="$TB_LOGDIR" --port=$TB_PORT --bind_all
            elif python3 -m tensorboard.main --help >/dev/null 2>&1; then
                python3 -m tensorboard.main --logdir="$TB_LOGDIR" --port=$TB_PORT --bind_all
            else
                echo "ERROR: TensorBoard binary or python module not found on $(hostname)" >&2
                exit 1
            fi
        ' >&2
    ) &
    TB_PID=$!
    
    # Register exit trap to ensure TensorBoard is killed once the job finishes
    _tb_cleanup() {{
        echo "Cleaning up TensorBoard server..." >&2
        if [ -n "$TB_PID" ]; then
            kill $TB_PID 2>/dev/null
            pkill -P $TB_PID 2>/dev/null
        fi
        pkill -u $USER -f "tensorboard.*--port=$TB_PORT" 2>/dev/null
    }}
    trap _tb_cleanup EXIT
    
    echo "TensorBoard auto-started with srun on node $NODE_NAME port $TB_PORT" >&2
fi
"""

