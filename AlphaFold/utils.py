import os
import json
import subprocess
from datetime import date
from collections import defaultdict
from pathlib import Path

from drona_utils import drona_add_additional_file, drona_add_mapping, drona_add_message
from drona_runtime_utils import drona_cluster_name, retrieve_slurm_params

aces_parafold_modules = '''module load GCC/11.3.0  OpenMPI/4.1.4 AlphaFold/2.3.2-CUDA-11.8.0
module load ParaFold/2.0-CUDA-11.8.0'''

grace_parafold_modules = '''module load GCC/11.3.0  OpenMPI/4.1.4
module load ParaFold/2.0-CUDA-11.7.0'''

faster_parafold_modules = '''module load GCC/11.3.0  OpenMPI/4.1.4 
module load ParaFold/2.0-CUDA-11.7.0'''

aces_pickle_module = '''module load AlphaPickle/1.4.1'''

grace_pickle_module = ''' module load AlphaPickle/1.4.1 '''

faster_pickle_module = '''module load AlphaPickle/1.4.1'''

def _value(v):
    """Field value, or "" when the field is unset (None, "" or the unmapped "$name")."""
    if v is None:
        return ""
    v = str(v).strip()
    return "" if v.startswith("$") else v


def _record_version(version):
    """Save the AlphaFold version in the workflow directory, so Manage mode knows which panels to show."""
    drona_add_additional_file(content=version + "\n", name=".drona/af_version", preview_order=-1)


def retrieve_driver_contents(mode, cancel_jobs=""):
    """Create mode submits the job files listed in [JOBFILES] with the shared sbatch driver;
    manage mode runs the cancel/delete actions."""
    mode = _value(mode)
    if mode == "":
        return 'echo "No workflow mode selected: choose Create or Manage." >&2\nexit 1\n'
    name = "driver-create.sh" if mode == "create" else "driver-manage.sh"
    return (Path(__file__).resolve().parent / "drivers" / name).read_text()


message_parafold='''
This script will run AlphaFold in two jobs, one for the CPU step (using Alphafold) and a second (using Parafold)  for the GPU steps and graph .pkl files.
You can find the respective scripts in the "Alphafold part" tab and the "Parafold part" tab in  this preview-pane.   
'''

message_reduced='''
This script will run AlphaFold in a single step using Alphafold. See the Reduced_dbs tab for the Alphafold command. You are welcome to make any changes before submitting.
'''

# AF3 attention settings per GPU (Triton on Ampere and newer, XLA on CUDA capability 7.x) are chosen at run
# time in jobscripts/alphafold3-gpu.job from the GPU the job got, so "Any GPU" works.


# Approximate AF3 input size each GPU handles without spilling to host memory (tokens). 80 GB values are
# from upstream docs/performance.md (5,120 tokens on A100/H100 80 GB); the others are estimates scaled by
# GPU memory. Larger inputs still run thanks to unified memory, but much slower.
AF3_GPU_TOKEN_LIMITS = {
    "h100": (5120, "80 GB"), "a100": (5120, "80 GB"),
    "a40": (3000, "48 GB"),
    "a30": (1500, "24 GB"), "a10": (1500, "24 GB"), "rtx": (1500, "24 GB"),
    "t4": (1000, "16 GB"),
}
# CCD ligands: size unknown without the CCD dictionary; ~30 heavy atoms (ATP 31, NAG 14, HEM 43).
_AF3_CCD_LIGAND_TOKENS = 30


def _ids(entry):
    ids = entry.get("id", "A")
    return len(ids) if isinstance(ids, list) else 1


def _smiles_atoms(smiles):
    """Heavy atoms in a SMILES string (bracket atoms + organic subset), hydrogens excluded."""
    import re
    atoms = re.findall(r"\[[^\]]*\]|Br|Cl|[BCNOPSFI]|[bcnops]", smiles or "")
    return sum(1 for a in atoms if not re.match(r"\[\d*H[\d+\-\]]", a))


def af3_token_count(path):
    """Approximate AF3 token count of an input JSON (alphafold3 or alphafoldserver dialect), or None.
    Polymers: 1 token per residue/nucleotide per copy; ligands: 1 token per heavy atom (CCD ligands
    estimated). Modified residues and ions are counted as one token each."""
    try:
        with open(os.path.expanduser(path)) as f:
            data = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    tokens = 0
    if isinstance(data, list):   # alphafoldserver dialect: list of jobs, entries with "count"
        data = data[0] if data else {}
        for seq in data.get("sequences", []):
            for kind, entry in seq.items():
                if not isinstance(entry, dict):
                    continue
                copies = int(entry.get("count", 1) or 1)
                if "sequence" in entry:
                    tokens += len(entry["sequence"]) * copies
                else:
                    tokens += _AF3_CCD_LIGAND_TOKENS * copies if kind == "ligand" else copies
        return tokens
    for seq in data.get("sequences", []):
        for kind, entry in seq.items():
            if not isinstance(entry, dict):
                continue
            copies = _ids(entry)
            if kind in ("protein", "rna", "dna"):
                tokens += len(entry.get("sequence", "")) * copies
            elif kind == "ligand":
                if entry.get("smiles"):
                    tokens += _smiles_atoms(entry["smiles"]) * copies
                else:
                    tokens += _AF3_CCD_LIGAND_TOKENS * len(entry.get("ccdCodes", [])) * copies
    return tokens


def _af3_size_hint(tokens, gpu_type):
    """Preview note/warning about the AF3 input size against the chosen GPU."""
    if tokens is None:
        return
    gpu_type = (gpu_type or "").lower()
    size = "AlphaFold 3 input: about %d tokens" % tokens
    if gpu_type == "any":
        known = sorted((AF3_GPU_TOKEN_LIMITS[g][0], g) for g in _cluster_gpu_types() if g in AF3_GPU_TOKEN_LIMITS)
        if not known:
            drona_add_message(size + ".", "note")
        elif tokens <= known[0][0]:
            drona_add_message("%s; fits on every GPU type (smallest: %s, up to ~%d tokens)."
                              % (size, known[0][1].upper(), known[0][0]), "note")
        else:
            small = [g.upper() for l, g in known if l < tokens]
            big = [g.upper() for l, g in known if l >= tokens]
            drona_add_message("%s. Any GPU may start on %s, which holds less (the job then spills to host memory "
                              "and runs much slower, or may fail).%s"
                              % (size, "/".join(small), (" Select %s to be safe." % big[0]) if big else ""),
                              "warning")
        return
    limit, mem = AF3_GPU_TOKEN_LIMITS.get(gpu_type, (None, None))
    if limit is None:
        drona_add_message(size + ".", "note")
    elif tokens <= limit:
        drona_add_message("%s; fits on %s (%s, up to ~%d tokens)." % (size, gpu_type.upper(), mem, limit), "note")
    else:
        bigger = sorted((l, g) for g, (l, _) in AF3_GPU_TOKEN_LIMITS.items()
                        if l >= tokens and g in _cluster_gpu_types())
        hint = (" Select %s instead." % bigger[0][1].upper()) if bigger else ""
        drona_add_message("%s; more than fits on %s (%s, up to ~%d tokens). The job spills to host memory and runs "
                          "much slower, or may fail.%s" % (size, gpu_type.upper(), mem, limit, hint), "warning")


def _cluster_gpu_types():
    return {g.lower() for g in CLUSTER_GPU_TYPES.get(drona_cluster_name(), set())}

def retrieve_modelpreset(modelpreset=""):
    if modelpreset == "":
        drona_add_mapping("modelpreset","monomer")
    else:
        drona_add_mapping("modelpreset",modelpreset)


def retrieve_dbpreset(db_preset=""):
    if db_preset == "reduced_dbs":
        drona_add_message("Parafold not available for reduced_dbs yet. Using single Alphafold step","warning")
        drona_add_additional_file("jobscripts/reduced_dbs.job", "Reduced_dbs")
        drona_add_mapping("JOBFILES", "reduced_dbs.job")
        drona_add_mapping("message",message_reduced)
    else:
        drona_add_additional_file("jobscripts/parafold-cpu.job", "parafold-cpu")
        drona_add_additional_file("jobscripts/parafold-gpu.job", "parafold-gpu")
        # CPU job first; the shared sbatch driver makes the GPU job depend on it
        drona_add_mapping("JOBFILES", "parafold-cpu.job parafold-gpu.job")
        drona_add_mapping("message",message_parafold)
        if db_preset == "":
            db_preset="full_dbs"
    drona_add_mapping("dbpreset",db_preset)

def set_modules_and_pickle(pickle, outputdir="parafold_output_dir"):
    output = drona_cluster_name()
    if output == "aces":
        drona_add_mapping("ALPHAFOLDMODULES",aces_parafold_modules )
        drona_add_mapping("ALPHAPICKLEMODULE",aces_pickle_module )
    elif output == "grace":
        drona_add_mapping("ALPHAFOLDMODULES",grace_parafold_modules )
        drona_add_mapping("ALPHAPICKLEMODULE",grace_pickle_module )
        drona_add_message("AlphaPickle not installed on Grace.","warning")
    elif output == "faster":
        drona_add_mapping("ALPHAFOLDMODULES",faster_parafold_modules )
        drona_add_mapping("ALPHAPICKLEMODULE",faster_pickle_module )
    else:
        drona_add_mapping("ALPHAFOLDMODULES","DUMMY MODULES : "+output+" :")
        drona_add_mapping("ALPHAPICKLEMODULE","DUMMY PICKLE" )
    snippet=""
    # The checkbox sends "true" when ticked (older forms sent "Yes").
    if str(pickle).strip().lower() in ("yes", "true", "on", "1") and output != "grace":
        # AlphaPickle reads ranking_debug.json + result_model_*.pkl from the per-protein
        # result directory (run_alphafold.sh names it after the FASTA file) and writes its plots there.
        snippet = ("# AlphaPickle: pLDDT plots (plus PAE plots for monomer_ptm / multimer)\n"
                   "af_result_dir=" + (outputdir or "parafold_output_dir") + "/$(basename \"${protein_fasta%.*}\")\n"
                   "run_AlphaPickle.py -od \"$af_result_dir\"")
    drona_add_mapping("pickle",snippet)

# GPU types AlphaFold can use per cluster (for the AF3 size hint; keep in step with retriever_scripts/retrieve_gpus)
CLUSTER_GPU_TYPES = {
    "aces": {"A30", "H100"},
    "grace": {"T4", "A40", "RTX"},
    "faster": {"T4", "A10", "A30", "A40"},
}


def _resolve_gpu(gpu):
    """Lower-case GPU type for the job. "", "any" and the old "auto" mean any GPU: an untyped request
    (cluster profile), so the job starts on whichever GPU is free first."""
    gpu = _value(gpu).lower()
    return "any" if gpu in ("", "any", "auto") else gpu


# GPU job resources per pipeline: (cpus per task, memory, walltime field used)
GPU_STAGE = {
    "alphafold3": (8, "64G", "gpu"),
    "full_dbs": (24, "122G", "gpu"),
    "reduced_dbs": (24, "180G", "cpu"),   # single job: uses the first "Expected time" field
}


def retrieve_af_slurm_params(mode, version, dbpreset, gpu, walltime, gpuwalltime, account, extra, jsoninput=""):
    """Slurm settings for the AlphaFold jobs.

    The GPU job (the only job for reduced_dbs) goes through the shared cluster checks
    (retrieve_slurm_params): TIME/NODES/CPUS/MEM/PARTITION/EXTRA. The CPU job keeps
    AlphaFold's fixed resources and gets CPUTIME; it uses the same EXTRA (account and
    additional Slurm parameters)."""
    if _value(mode) != "create":
        return ""   # Manage mode submits no jobs; nothing to do
    version = _value(version)
    if version not in ("alphafold2", "alphafold3"):
        return ""
    stage = "alphafold3" if version == "alphafold3" else ("reduced_dbs" if _value(dbpreset) == "reduced_dbs" else "full_dbs")
    cpus, mem, time_field = GPU_STAGE[stage]

    cpu_time = _value(walltime) or "48:00"
    gpu_time = cpu_time if time_field == "cpu" else (_value(gpuwalltime) or "24:00")
    drona_add_mapping("CPUTIME", cpu_time + ":00")

    tokens = None
    if version == "alphafold3" and _value(jsoninput):
        tokens = af3_token_count(_value(jsoninput))
    gpu_type = _resolve_gpu(gpu)
    if version == "alphafold3":
        _af3_size_hint(tokens, gpu_type)

    return retrieve_slurm_params("1", "1", str(cpus), mem, gpu_type, "1", gpu_time,
                                 _value(account), _value(extra))


def retrieve_maxtemplate(maxtemplate=""):
    if maxtemplate == "":
        return f""+str(date.today())
    else:
        return f""+maxtemplate

def retrieve_datadir_vars(datadir=""):
    datadirstring="/scratch/data/bio/alphafold/2.3.2"
    if datadir != "":
        datadirstring=datadir
    drona_add_mapping("datadir",datadirstring)
             
             
def retrieve_fasta(proteinfasta=""):
    if proteinfasta=="":
        drona_add_message("ERROR: No FASTA file specified. Your job will not run. Please cancel and select a FASTA file or edit the preview files directly.","error")
    drona_add_mapping("proteinfasta",proteinfasta)


AF3_DEFAULT_RECYCLES = 10  # run_alphafold.py default


def _af3_recycles(recycles):
    """--num_recycles value: the form value, or the AlphaFold 3 default when empty/invalid."""
    try:
        n = int(float(str(recycles).strip()))
    except (TypeError, ValueError):
        return str(AF3_DEFAULT_RECYCLES)
    if n < 1:
        drona_add_message("Recycles must be at least 1; using the AlphaFold 3 default of %d." % AF3_DEFAULT_RECYCLES, "warning")
        return str(AF3_DEFAULT_RECYCLES)
    return str(n)


def _fasta_record_count(path):
    """Number of '>' records in a FASTA file, or None when it cannot be read (e.g. not uploaded yet)."""
    try:
        with open(path, errors="replace") as f:
            return sum(1 for line in f if line.startswith(">"))
    except OSError:
        return None


def _check_fasta_preset(proteinfasta, modelpreset):
    """Preview check: the multimer preset needs 2+ chains, the monomer presets exactly one.
    A mismatch would otherwise only fail after the CPU stage (hours)."""
    proteinfasta = _value(proteinfasta)
    preset = _value(modelpreset) or "monomer"
    if not proteinfasta:
        return
    n = _fasta_record_count(proteinfasta)
    if n is None:
        return
    if n == 0:
        drona_add_message("The FASTA file " + proteinfasta + " has no sequences ('>' header lines); the job will fail.", "error")
    elif preset == "multimer" and n < 2:
        drona_add_message("The multimer preset is meant for complexes, but " + proteinfasta + " has one sequence, so a "
                          "single chain is predicted. For a homo-oligomer, repeat the sequence once per copy; "
                          "for a single chain, monomer_ptm is faster.", "warning")
    elif preset != "multimer" and n > 1:
        drona_add_message("The FASTA file has %d sequences, but the %s preset predicts a single chain and will fail. "
                          "Use the multimer preset to predict the complex." % (n, preset), "error")


def retrieve_mail(notification="", email=""):
    """#SBATCH line for mail notifications, or "" when none are requested."""
    notification = _value(notification)
    email = _value(email)
    if notification in ("", "none"):
        return ""
    result = "#SBATCH --mail-type=" + notification
    if email:
        result += " --mail-user=" + email
    return result

AF2_CPU_STAGE_DEFAULTS = (8, 488)   # cores, GB: AlphaFold's searches use at most 8 threads


def _positive_int(value, default, label):
    """Whole number >= 1 from a form field, or the default (with an error message when invalid)."""
    value = _value(value)
    if not value:
        return default
    try:
        n = int(float(value))
    except ValueError:
        n = 0
    if n < 1:
        drona_add_message("%s must be a whole number of at least 1; using %d." % (label, default), "error")
        return default
    return n


def af2_cpu_stage_resources(cores, mem):
    """CPU job (full_dbs MSA/template search) cores and memory from ADVANCED."""
    cores = _positive_int(cores, AF2_CPU_STAGE_DEFAULTS[0], "CPU stage cores")
    mem = _positive_int(mem, AF2_CPU_STAGE_DEFAULTS[1], "CPU stage memory")
    if cores > 8:
        drona_add_message("CPU stage: %d cores requested, but AlphaFold's searches use at most 8 threads; the "
                          "extra cores stay idle." % cores, "note")
    drona_add_mapping("AF2CPUS", str(cores))
    drona_add_mapping("AF2MEM", "%dG" % mem)


def process_alphafold2(version,dbpreset,proteinfasta,modelpreset,datadir,pickle,outputdir,cpu_cores="",cpu_mem=""):
    if version != "alphafold2":
        return ""
    _record_version("alphafold2")
    retrieve_modelpreset(modelpreset)
    retrieve_dbpreset(dbpreset)
    if _value(dbpreset) != "reduced_dbs":
        af2_cpu_stage_resources(cpu_cores, cpu_mem)
    set_modules_and_pickle(pickle, outputdir)
    retrieve_fasta(proteinfasta)
    _check_fasta_preset(proteinfasta, modelpreset)
    retrieve_datadir_vars(datadir)
    return f""



# Starting point for the alphafold3-input.json tab when no input file is chosen. The placeholder
# sequence is not a valid protein sequence, so an unedited skeleton fails at the start of the CPU job.
AF3_INPUT_SKELETON = {
    "name": "",
    "sequences": [
        {"protein": {"id": "A", "sequence": "REPLACE_WITH_PROTEIN_SEQUENCE"}}
    ],
    "modelSeeds": [1],
    "dialect": "alphafold3",
    "version": 1,
}


def _af3_input_json(jsoninput, name):
    """Content of alphafold3-input.json: the chosen input file, or a skeleton to edit in the preview."""
    path = (jsoninput or "").strip()
    if not path:
        drona_add_message("No input JSON selected. Edit the example in the alphafold3-input.json tab of the "
                          "preview (replace the sequence), or select an input file. The input is checked again when you submit.", "note")
        return json.dumps(dict(AF3_INPUT_SKELETON, name=name), indent=2) + "\n"
    path = os.path.expanduser(path)
    try:
        with open(path) as f:
            content = f.read()
    except OSError as e:
        drona_add_message(f"Cannot read the input JSON {path}: {e.strerror or e}. Your job will not run.", "error")
        return ""
    try:
        json.loads(content)
    except ValueError as e:
        drona_add_message(f"The input JSON {os.path.basename(path)} is not valid JSON ({e}). "
                          "Fix it in the alphafold3-input.json tab or select another file.", "warning")
    return content


def _seed_number(seeds, prefix=""):
    """Validated --num_seeds value: "" when empty, the number as a string, or None after an error."""
    seeds = _value(seeds)
    if not seeds:
        return ""
    try:
        n = int(float(seeds))
    except ValueError:
        drona_add_message(prefix + "Seeds must be a whole number.", "error")
        return None
    if n < 1:
        drona_add_message(prefix + "Seeds must be at least 1.", "error")
        return None
    return str(n)


def _listed_seeds(content):
    """Number of modelSeeds in AF3 input JSON text, or None when it cannot be parsed."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return None
    if isinstance(data, list):   # alphafoldserver dialect: seeds are chosen by AlphaFold itself
        return None
    return len(data.get("modelSeeds") or []) if isinstance(data, dict) else None


def _create_seeds(seeds, input_content):
    """" --num_seeds=N" for the GPU job, or "". Error when the input already lists several seeds,
    which AlphaFold 3 rejects together with --num_seeds."""
    n = _seed_number(seeds)
    if not n:
        return ""
    listed = _listed_seeds(input_content)
    if listed is not None and listed > 1:
        drona_add_message("Seeds: the input already lists %d seeds (modelSeeds); AlphaFold 3 can only add seeds "
                          "when the input has one. Leave Seeds empty, or keep one seed in the input." % listed, "error")
        return ""
    if n != "1":
        drona_add_message("Seeds: %s seeds x 5 samples; the GPU stage takes about %s times as long as with one seed."
                          % (n, n), "note")
    return " --num_seeds=" + n


def af3_submit_check(mode, version):
    """Create-driver lines that check alphafold3-input.json (as submitted, with preview edits) before sbatch."""
    if _value(mode) != "create" or _value(version) != "alphafold3":
        return ""
    import shlex
    helper = shlex.quote(str(Path(__file__).resolve().parent / "check_af3_input.py"))
    return ("# Check the AlphaFold 3 input before anything is queued\n"
            "python3 %s alphafold3-input.json alphafold3-gpu.job || exit 1" % helper)


def process_alphafold3(version,jsoninput,modelinput,recycles,location,af3name="",seeds=""):
    if version != "alphafold3":
        return ""

    # process alphafold3 params
    drona_add_additional_file("jobscripts/alphafold3-cpu.job","alphafold3-cpu.job")
    drona_add_additional_file("jobscripts/alphafold3-gpu.job","alphafold3-gpu.job")
    drona_add_additional_file("jobscripts/alphafold3-input.json","alphafold3-input.json")
    drona_add_mapping("JOBFILES", "alphafold3-cpu.job alphafold3-gpu.job")
    _record_version("alphafold3")

    drona_add_mapping("RECYCLES", _af3_recycles(recycles))
    drona_add_mapping("JSONDIR",location)
    drona_add_mapping("JSONFILE","alphafold3-input.json")

    # Run name — sanitise to safe characters, default to "af3run"
    name = (af3name or "").strip()
    if not name:
        name = "af3run"
    import re as _re
    name = _re.sub(r'[^A-Za-z0-9_-]', '_', name)
    drona_add_mapping("AF3NAME", name)

    # Output directory — use the job's working directory (from location picker)
    drona_add_mapping("AF3OUTPUTDIR", location)


    input_content = _af3_input_json(jsoninput, name)
    drona_add_mapping("JSONINPUT", input_content)
    drona_add_mapping("SEEDS", _create_seeds(seeds, input_content))
    drona_add_mapping("ALPHAFOLD3MODEL", _af3_model_dir(modelinput))
    return ""


_AF3_MODEL_FILES = ("af3.bin", "af3.bin.zst")


def _af3_model_dir(modelinput):
    """Return the AF3 model parameters directory; add a preview error when it
    does not contain af3.bin(.zst). A picked weights file is accepted too."""
    modelinput = (modelinput or "").strip().rstrip("/")
    if not modelinput:
        drona_add_message("No AlphaFold 3 model parameters directory selected; "
                          "the GPU job will not run.", "error")
        return ""
    if os.path.isfile(modelinput) and os.path.basename(modelinput) in _AF3_MODEL_FILES:
        return os.path.dirname(modelinput)
    if not os.path.isdir(modelinput):
        drona_add_message("AlphaFold 3 model parameters directory " + modelinput +
                          " does not exist; the GPU job will not run.", "error")
        return modelinput
    if any(os.path.isfile(os.path.join(modelinput, f)) for f in _AF3_MODEL_FILES):
        return modelinput
    # Weights one level down (e.g. the folder the download was unpacked into)?
    hints = []
    try:
        for entry in sorted(os.listdir(modelinput)):
            sub = os.path.join(modelinput, entry)
            if os.path.isdir(sub) and any(os.path.isfile(os.path.join(sub, f)) for f in _AF3_MODEL_FILES):
                hints.append(sub)
    except OSError:
        pass
    msg = ("No af3.bin or af3.bin.zst in " + modelinput +
           "; the GPU job would fail with 'No models matched'.")
    if hints:
        msg += " Found the weights in " + ", ".join(hints) + " - select that directory instead."
    drona_add_message(msg, "error")
    return modelinput


# ── Manage: re-run the GPU stage (see rerun_gpu_stage.py and md-files/alphafold-rerun-gpu-stage-design.md) ──
GPU_JOB_FILES = {"alphafold3": "alphafold3-gpu.job", "alphafold2": "parafold-gpu.job"}


def _manage_dir(job_dir):
    """drona_job_dir may arrive as a plain path or as an HTML link around it."""
    import re
    job_dir = _value(job_dir)
    m = re.search(r'>(/[^<]+)<', job_dir)
    return (m.group(1) if m else job_dir).strip()


def rerun_gpu_stage(mode, enable, version, job_dir, wf_id, rerun_state, gpu, walltime, recycles, modeldir, seeds=""):
    """Manage-mode shell snippet that re-runs the GPU stage on the finished CPU output, or "".

    Checked at preview: the workflow has a finished CPU stage and no active GPU job, the job file
    exists, and the overrides are valid. The snippet calls rerun_gpu_stage.py, which backs up and
    edits the job file, submits it and updates the workflow record."""
    import shlex
    if _value(mode) != "manage" or _value(enable).lower() not in ("yes", "true", "on", "1"):
        return ""
    version = _value(version)
    state = _value(rerun_state)
    wf_id = _value(wf_id)
    workdir = _manage_dir(job_dir)
    jobfile = GPU_JOB_FILES.get(version)
    if not jobfile or not wf_id:
        drona_add_message("Re-run: select an AlphaFold 2 or 3 workflow first.", "error")
        return ""
    if state not in ("failed", "done"):
        drona_add_message("Re-run: the GPU stage can only be re-run when the CPU stage has completed and the "
                          "GPU job is no longer pending or running.", "error")
        return ""
    if not os.path.isfile(os.path.join(workdir, jobfile)):
        drona_add_message("Re-run: " + jobfile + " not found in " + workdir + ".", "error")
        return ""
    if os.path.exists(os.path.join(workdir, ".drona", "af_cleaned")):
        drona_add_message("Re-run: the intermediate files of this run were removed (FREE UP SPACE), and the GPU "
                          "stage needs them. Create a new workflow to predict again.", "error")
        return ""
    if version == "alphafold2" and state == "done":
        drona_add_message("Re-run: the AlphaFold 2 GPU stage has completed. Running it again with the same input "
                          "gives the same models, so there is nothing to re-run.", "error")
        return ""

    args = ["--workflow", wf_id, "--jobfile", jobfile]

    walltime = _value(walltime)
    gpu = _value(gpu).lower()
    if walltime or gpu not in ("", "keep"):
        stage = "alphafold3" if version == "alphafold3" else "full_dbs"
        cpus, mem, _ = GPU_STAGE[stage]
        if gpu in ("", "keep"):
            gpu_type = _current_gpu(os.path.join(workdir, jobfile))
        else:
            gpu_type = _resolve_gpu(gpu)
        captured = {}

        def capture(key, value):
            captured[key] = value
            return value
        retrieve_slurm_params("1", "1", str(cpus), mem, gpu_type, "1", walltime or "24:00", "", "",
                              mapping_filter=capture)
        if walltime and captured.get("TIME"):
            args += ["--time", captured["TIME"]]
        if gpu not in ("", "keep") and captured.get("PARTITION"):
            args += ["--partition", captured["PARTITION"]]

    if version == "alphafold3":
        current = _af3_job_settings(os.path.join(workdir, jobfile))
        changes = []
        if _value(recycles):
            n = _af3_recycles(recycles)
            args += ["--recycles", n]
            if n != current.get("recycles"):
                changes.append("recycles")
        if _value(modeldir):
            d = _af3_model_dir(modeldir)
            args += ["--modeldir", d]
            if os.path.realpath(d) != os.path.realpath(current.get("modeldir") or "/nonexistent"):
                changes.append("model parameters")
        n_seeds = _af3_seeds(seeds, workdir)
        if n_seeds is None:
            return ""
        if n_seeds:
            args += ["--seeds", n_seeds]
            if n_seeds != current.get("seeds"):
                changes.append("seeds")
        if state == "done" and not changes:
            drona_add_message("Re-run: the GPU stage has completed. With the same recycles, seeds and model "
                              "parameters the prediction is identical; change at least one of them.", "error")
            return ""

    helper = str(Path(__file__).resolve().parent / "rerun_gpu_stage.py")
    cmd = "python3 " + " ".join(shlex.quote(a) for a in [helper] + args)
    return ('echo "==> Re-running the GPU stage of workflow %s"\n'
            'cd %s && %s\n' % (wf_id, shlex.quote(workdir), cmd))


def _current_gpu(jobfile):
    """GPU type from the job file's --gres=gpu:<type>:<n> ("any" for an untyped --gres=gpu:<n>),
    for a walltime-only re-run."""
    import re
    try:
        with open(jobfile) as f:
            text = f.read()
        m = re.search(r"--gres=gpu:([A-Za-z][A-Za-z0-9_]*):\d+", text)
        if m:
            return m.group(1).lower()
        return "any" if re.search(r"--gres=gpu:\d+", text) else ""
    except OSError:
        return ""


def _af3_job_settings(jobfile):
    """Current AF3 GPU-stage settings in the job file: recycles, seeds (--num_seeds) and model directory."""
    import re
    try:
        with open(jobfile) as f:
            text = f.read()
    except OSError:
        return {}
    out = {}
    m = re.search(r"--num_recycles=(\d+)", text)
    if m:
        out["recycles"] = m.group(1)
    m = re.search(r"--num_seeds=(\d+)", text)
    if m:
        out["seeds"] = m.group(1)
    m = re.search(r"^AF3_MODEL_PARAMETERS_DIR=(.*)$", text, re.M)
    if m:
        out["modeldir"] = m.group(1).strip()
    return out


def _af3_seeds(seeds, workdir):
    """--num_seeds value ("" = not set), or None after an error message.
    AlphaFold 3 only accepts --num_seeds when the input (the CPU stage's <name>_data.json) lists one seed."""
    import glob
    n = _seed_number(seeds, "Re-run: ")
    if not n:
        return n
    data = []
    try:
        with open(os.path.join(workdir, "af3_output_dir.txt")) as f:
            base = f.read().strip()
        data = sorted(glob.glob(os.path.join(base, "*", "*_data.json")))
    except OSError:
        pass
    if data:
        try:
            with open(data[0]) as f:
                listed = json.load(f).get("modelSeeds") or []
        except (OSError, ValueError):
            listed = []
        if len(listed) > 1:
            drona_add_message("Re-run: the input already lists %d seeds (modelSeeds); AlphaFold 3 can only add seeds "
                              "when the input has one. Leave Seeds empty." % len(listed), "error")
            return None
    return n


def af_auto_bundle(mode, version):
    """Job-file lines that zip the results for download when the prediction succeeded, so a finished run
    always has a download link (Manage's Create bundle remakes it after a re-run)."""
    version = _value(version)
    if _value(mode) != "create" or version not in ("alphafold2", "alphafold3"):
        return ""
    import shlex
    helper = shlex.quote(str(Path(__file__).resolve().parent / "make_af_bundle.py"))
    return ("# Download bundle (structures, scores, plots) for the Manage page\n"
            "if [ $af_status -eq 0 ]; then\n"
            "    python3 %s \"$SLURM_SUBMIT_DIR\" %s || echo \"Download bundle not created; use Create bundle in Manage.\"\n"
            "fi" % (helper, version))


def clean_intermediates(mode, enable, version, job_dir, status=""):
    """Manage-mode shell snippet that removes the run's large intermediate files (clean_af_intermediates.py),
    or "". Only for finished runs; the preview lists what will be removed."""
    import shlex
    if _value(mode) != "manage" or _value(enable).lower() not in ("yes", "true", "on", "1"):
        return ""
    version = _value(version)
    if version not in ("alphafold2", "alphafold3"):
        drona_add_message("Free up space: select an AlphaFold 2 or 3 workflow first.", "error")
        return ""
    if _value(status) != "DONE":
        drona_add_message("Free up space: only possible after the run has finished successfully.", "error")
        return ""
    workdir = _manage_dir(job_dir)
    helper = str(Path(__file__).resolve().parent / "clean_af_intermediates.py")
    try:
        listing = json.loads(subprocess.check_output(["python3", helper, workdir, version, "--list"],
                                                     universal_newlines=True, timeout=60))
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        drona_add_message("Free up space: could not list the intermediate files (%s)." % e, "error")
        return ""
    files = listing.get("files", [])
    if not files:
        drona_add_message("Free up space: no intermediate files left in this run.", "error")
        return ""
    import fnmatch
    names = (os.path.basename(f) for f, _ in files)
    kinds = sorted({"result_model_*.pkl" if fnmatch.fnmatch(n, "result_model_*.pkl") else n for n in names})
    drona_add_message("Free up space: Run removes %d item(s), %.1f MB (%s). This cannot be undone, and the GPU "
                      "stage can no longer be re-run. Structures, scores, plots and the download bundle are kept."
                      % (len(files), listing.get("bytes", 0) / 1e6, ", ".join(kinds[:6])), "warning")
    return ('echo "==> Removing intermediate files"\n'
            "python3 " + " ".join(shlex.quote(a) for a in [helper, workdir, version]))


def make_bundle(mode, enable, version, job_dir, status=""):
    """Manage-mode shell snippet that zips the finished results for download (make_af_bundle.py), or ""."""
    import shlex
    if _value(mode) != "manage" or _value(enable).lower() not in ("yes", "true", "on", "1"):
        return ""
    version = _value(version)
    if version not in ("alphafold2", "alphafold3"):
        drona_add_message("Download bundle: select an AlphaFold 2 or 3 workflow first.", "error")
        return ""
    if _value(status) and _value(status) != "DONE":
        drona_add_message("Download bundle: the workflow has not finished; the bundle holds the results "
                          "available now.", "warning")
    workdir = _manage_dir(job_dir)
    helper = str(Path(__file__).resolve().parent / "make_af_bundle.py")
    return ('echo "==> Creating the download bundle"\n'
            "python3 " + " ".join(shlex.quote(a) for a in [helper, workdir, version]))
