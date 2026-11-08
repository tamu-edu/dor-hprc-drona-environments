# AlphaFold (Drona environment)

Runs AlphaFold 2 (ParaFold: CPU job for the genetic search, then GPU job for the prediction,
or one GPU job with the reduced databases) and AlphaFold 3 (CPU data pipeline + GPU inference).
The Manage page shows the job timeline, a results summary, the structure/PAE viewers, and can
re-run the GPU stage.

## Layout

- `schema.json`, `schemas/`: create and manage forms; `map.json`, `utils.py`: preview logic
- `jobscripts/`, `drivers/`: Slurm job files and the create/manage driver scripts
- `retriever_scripts/`: retrievers for the form and the Manage cards (referenced as
  `retriever_scripts/<name>` in the schemas). `list_accounts` stays in the top directory because the
  shared account field (`drona_slurm_resources.json`) looks it up there by name.
  `retriever_scripts/af_results.py` holds the shared lookups (newest AF2 result / AF3 prediction directory,
  no recursive globs) and the viewer matrix shrinking (max 500 x 500 cells, block maximum).
- `rerun_gpu_stage.py`, `make_af_bundle.py`: helpers the Manage actions run; `af_viewer.js`, `af3_viewer.js`:
  result viewers

## Porting to another cluster

The workflow logic is not cluster-specific. Slurm partitions, accounts and walltime limits
come from the Drona cluster profile (`runtime_support/.../clusters/<cluster>.py` in the
composer), and the cluster name from `drona_cluster_name()` (`DRONA_CLUSTER`, else Slurm's
`ClusterName`). What a new cluster needs are its **module names, data paths and GPU types**,
which today live in the files below.

Cluster names in the code are lower-case, as returned by `drona_cluster_name()`
(e.g. `aces`, `grace`, `faster`).

### 1. AlphaFold 2 / ParaFold modules: `utils.py`

- Add `<cluster>_parafold_modules` (the `module load` lines for AlphaFold 2 + ParaFold) and
  `<cluster>_pickle_module` (AlphaPickle) next to the existing ones at the top of the file.
- Add an `elif output == "<cluster>":` branch in `set_modules_and_pickle()` that maps both.
  Unknown clusters get `DUMMY MODULES` in the job file, so the preview shows the gap.
- No AlphaPickle on the cluster: copy the Grace branch (warning message) and add the cluster to
  the `output != "grace"` condition that skips the AlphaPickle step.

The AF2 job files (`jobscripts/parafold-*.job`, `reduced_dbs.job`) use the
`[ALPHAFOLDMODULES]` / `[ALPHAPICKLEMODULE]` placeholders and need no change.

### 2. AlphaFold 3: `jobscripts/alphafold3-cpu.job` and `jobscripts/alphafold3-gpu.job`

Both files contain the same four settings. Change them in **both**:

- `module load AlphaFold3/<version>`
- `AF3_CODE_DIR`: AlphaFold 3 source folder
- `AF3_DATABASES_DIR`: AlphaFold 3 genetic databases
- `AF3_IMAGE`: AlphaFold 3 Singularity/Apptainer image

The model weights (`af3.bin` / `af3.bin.zst`) are per user (license) and picked in the form.

### 3. GPU types: `utils.py` and `retriever_scripts/retrieve_gpus`

The same list exists in two places. Keep them in step:

- `retriever_scripts/retrieve_gpus`: a `case` line per cluster with the GPU types for the form dropdown
  (`value:Label` pairs, value = lower-case Slurm GRES name). The labels add free GPUs (`sinfo`) and
  pending jobs per type (`squeue`).
- `utils.py` `CLUSTER_GPU_TYPES`: the same GPU types (upper-case), for the AF3 size hint.
- `utils.py` `AF3_GPU_TOKEN_LIMITS`: approximate AF3 input size (tokens) per GPU type, for the preview
  size hint. Add new GPU types.

The first dropdown entry, "Any GPU (first available)", sends `gpu = "any"` to the Drona cluster
profile, which requests an untyped GPU (`--gres=gpu:N`) on the partitions that hold GPUs (e.g. Grace:
`--partition=gpu,gpu-a40`). A new cluster's profile needs an `any` branch for this.

The AF3 GPU job chooses its attention settings at run time from the GPU it got
(`nvidia-smi --query-gpu=compute_cap`): XLA attention + the XLA workaround flag on CUDA capability
7.x, Triton flash attention otherwise. No per-GPU setting is needed.

The re-run card (`retriever_scripts/retrieve_rerun_gpus`) reuses `retriever_scripts/retrieve_gpus` and needs no change.

### 4. AlphaFold 2 databases and example files: `utils.py`, `schemas/create.schema.json`

- `utils.py`: default AF2 data folder (`datadirstring = "/scratch/data/bio/alphafold/2.3.2"`).
- `schemas/create.schema.json`, same folder again: the data-folder field's `default`, its
  `help` text and its picker "Examples" path.
- `schemas/create.schema.json`, the picker "Examples" folders for the AF2 FASTA
  (`/scratch/data/bio/alphafold/example_data/`) and AF3 input JSON
  (`/scratch/data/bio/alphafold3/examples/`). Remove the entry if the cluster has no examples.

### 5. AF3 viewer Python: `retriever_scripts/retrieve_af_combined.sh`

Sets `AF_BASE`, `LD_LIBRARY_PATH` and `PYTHONPATH` to HPRC EasyBuild paths (Python 3.10 +
SciPy-bundle). Point them at a Python 3 with NumPy on the new cluster (or replace the block
with the matching `module load`).

### 6. Accounts: `list_accounts`

Calls HPRC's `list_accounts.py`. Replace it with a command that prints the user's Slurm
accounts in the same format (see the composer's account dropdown), e.g. from `sacctmgr`.

### 7. Links to result files: `retriever_scripts/retrieve_af_pickle_plots.sh`, `retriever_scripts/retrieve_af_bundle.sh`

Plots and the download bundle are linked through the Open OnDemand Files app
(`FILES_URL = "/pun/sys/dashboard/files/fs"`). Change it if the site's dashboard URL differs.

### Check after porting

```bash
grep -rnE "aces|grace|faster|/sw/|/scratch/data|module load" . --exclude-dir=__pycache__
```

lists every remaining site-specific line. Then preview an AF2 and an AF3 job in the app: the
job files must show real module lines (no `DUMMY MODULES`) and the preview must not report
missing folders.
