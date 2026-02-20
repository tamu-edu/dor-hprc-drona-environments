import math
import os
import subprocess
import datetime
from collections import defaultdict

aces_parafold_modules = '''module load GCC/11.3.0  OpenMPI/4.1.4 AlphaFold/2.3.2-CUDA-11.8.0
module load ParaFold/2.0-CUDA-11.8.0'''

grace_parafold_modules = '''module load GCC/11.3.0  OpenMPI/4.1.4
module load ParaFold/2.0-CUDA-11.7.0'''

faster_parafold_modules = '''module load GCC/11.3.0  OpenMPI/4.1.4 
module load ParaFold/2.0-CUDA-11.7.0'''

aces_pickle_module = '''module load AlphaPickle/1.4.1'''

grace_pickle_module = ''' module load AlphaPickle/1.4.1 '''

faster_pickle_module = '''module load AlphaPickle/1.4.1'''

driver_alphafold3 = '''
out=`/sw/local/bin/sbatch alphafold3-cpu.job`
jobid=`echo $out | tail -n 1 | grep "Submitted batch job" | cut -d" " -f4`

if [ -z "${jobid}" ]; then
   echo "The Alphafold job was not submitted succesfully. Exiting now"
   exit 0
fi

echo "Submitted alphafold3-cpu.job, jobid: ${jobid}"
   
out=`/sw/local/bin/sbatch --dependency=afterok:${jobid} --kill-on-invalid-dep=yes  alphafold3-gpu.job`
jobidgpu=`echo $out | tail -n 1 | grep "Submitted batch job" | cut -d" " -f4`
echo
echo "Submitted alphafold3-gpu.job, jobid: ${jobidgpu} (dependency: completion job ${jobid})"
echo
echo "Submission part of your AlphaFold workflow has been completed. Once the requested resources are available, your job will start running."
echo
echo "NOTE: Do NOT hit the "Submit job" button  again. It will re-submit the worklow, using the same output directory"
'''

driver_parafold= '''
output1="--output=out-alphafold.%j --error=out-alphafold.%j"
aparams="--job-name=alphafold-[job-file-name] --time=[time] --ntasks=1 --cpus-per-task=48 --mem=488G [account] [email]"
jobid=`/sw/local/bin/sbatch ${aparams} ${output1} parafold-cpu.job  2> /dev/null | tail -n 1 | grep "Submitted batch job" | cut -d" " -f4`

if [ -z "${jobid}" ]; then
   echo "The Alphafold job was not submitted succesfully. Exiting now"
   exit 0
else
   output2="--output=out-parafold.%j --error=out-parafold.%j"
   pparams="--job-name=parafold-[job-file-name] --time=[gputime] --ntasks-per-node=1 --cpus-per-task=24 --mem=122G [gpu] [account] [email]"
   /sw/local/bin/sbatch ${pparams} ${output2}  --dependency=afterok:${jobid} parafold-gpu.job 
fi 
'''

driver_alphafold2='''
output="--output=out-alphafold.%j --error=error.alphafold.%j"
pparams="--time=[time] --ntasks-per-node=1 --cpus-per-task=24 --mem=180G [gpu] [account] [email]"

#submit the job
/sw/local/bin/sbatch --job-name=alphafold-[job-file-name] ${pparams} ${output}  reduced_dbs.job
'''


message_parafold='''
This script will run AlphaFold in two jobs, one for the CPU step (using Alphafold) and a second (using Parafold)  for the GPU steps and graph .pkl files.
You can find the respective scripts in the "Alphafold part" tab and the "Parafold part" tab in  this preview-pane.   
'''

message_reduced='''
his script will run AlphaFold in a single step using Alphafold. See the Reduced_dbs tab for the Alphafold command. You are welcome to make any changes before submitting.
'''

old_gpu_flags=''' \\  
  --nv --env 'XLA_PYTHON_CLIENT_PREALLOCATE=false' \\
  --env 'TF_FORCE_UNIFIED_MEMORY=true' \\
  --env 'XLA_CLIENT_MEM_FRACTION=3.2' \\
  --env XLA_FLAGS=--xla_disable_hlo_passes=custom-kernel-fusion-rewriter \\
  --flash_attention_implementation=xla
'''

def retrieve_modelpreset(modelpreset=""):
    if modelpreset == "":
        drona_add_mapping("modelpreset","monomer")
    else:
        drona_add_mapping("modelpreset",modelpreset)


def retrieve_dbpreset(db_preset=""):
    cwd=os.getcwd()
    if db_preset == "reduced_dbs":
        drona_add_message("Parafold not available for reduced_dbs yet. Using single Alphafold step","warning")
        drona_add_additional_file("reduced_dbs.job", "Reduced_dbs")
        drona_add_mapping("runcommand",driver_alphafold2)
        drona_add_mapping("message",message_reduced)
    else:
        drona_add_additional_file("parafold-gpu.job", "parafold-gpu")
        drona_add_additional_file("parafold-cpu.job", "parafold-cpu")
        drona_add_mapping("runcommand",driver_parafold)
        drona_add_mapping("message",message_parafold)
        if db_preset == "":
            db_preset="full_dbs"
    drona_add_mapping("dbpreset",db_preset)

def set_modules_and_pickle(pickle):
    output = subprocess.check_output(["/sw/local/bin/clustername"], text=True).strip()
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
    if pickle == "Yes" and output != "grace":
        snippet="#Run alphapickle  graph pLDDT and PAE .pkl files\nrun_AlphaPickle.py -od pickle_out_dir"
    drona_add_mapping("pickle",snippet)

def retrieve_gpu(gpu=""):
    
    finalgpustring=gpu
    if gpu == "" or gpu == "auto":

        cluster = subprocess.check_output(["/sw/local/bin/clustername"], text=True).strip()
        if cluster == "aces":
            gpu_set = {"A30", "H100"}
        elif cluster == "grace":
            gpu_set = {"T4", "A40", "RTX"}
        elif cluster == "faster":
            gpu_set = {"T4", "A10", "A30", "A40"}

        sums_by_gpu = defaultdict(int)

        output = subprocess.check_output(["/sw/local/bin/gpuavail","-m"], universal_newlines=True)
        for line in output.strip().split('\n'):
            try:
                count_part, gput_part = line.split('x', 1)
                total = int(count_part.strip())
                gpu_part = gput_part.split(':')[0].strip()

                # --- NEW: Check if T is in the allowed set ---
                # If a set of allowed values is provided and T is not in it,
                # skip to the next line.
                if gpu_part not in gpu_set:
                    continue

                sums_by_gpu[gpu_part] += total
            except (ValueError, IndexError):
                print(f"Warning: Skipping malformed line: '{line}'")
                continue

        highest_gpu = max(sums_by_gpu, key=sums_by_gpu.get)
        highest_sum = sums_by_gpu[highest_gpu]

        note_string = [f"{key}={value} " for key, value in sums_by_gpu.items()]
        final_string = ", ".join(note_string)
        drona_add_message("Available nodes with gpus: "+final_string+ "   -->  selecting "+highest_gpu, "note")
        
        finalgpustring=highest_gpu.lower()
    
    if finalgpustring == "rtx" or finalgpustring == "t4":
        drona_add_mapping("OLDGPUFLAGS", old_gpu_flags)
    else:
        drona_add_mapping("OLDGPUFLAGS", "")
    return f"--partition=gpu --gres=gpu:"+finalgpustring+":1"


def retrieve_maxtemplate(maxtemplate=""):
    if maxtemplate == "":
        return f""+str(datetime.date.today())
    else:
        return f""+maxtemplate

def retrieve_account(account=""):
    accountstring=""
    if account != "":
        accountstring="--account="+account
    return accountstring

def retrieve_datadir_vars(datadir=""):
    datadirstring="/scratch/data/bio/alphafold/2.3.2"
    if datadir != "":
        datadirstring=datadir
    drona_add_mapping("datadir",datadirstring)
             
             
def retrieve_fasta(proteinfasta=""):
    if proteinfasta=="":
        drona_add_message("ERROR: No FASTA file specified. Your job will not run. Please cancel and select a FASTA file or edit the preciew files directly.","error")
    drona_add_mapping("proteinfasta",proteinfasta)


def retrieve_mail(notification="", email=""):
    result = ""
    if not (notification == ""  or notification == "none"):
        result="--mail-type="+notification
        if email != "":
            result=result+" --mail-user="+email
    return result

def retrieve_walltime(time=""):
    if time == "":
        return "48:00:00" 
    else:
        times=time.split(':')
        total_hours = (int(times[0])+int(times[1])/60)
        if total_hours > 72:
            drona_add_message("Requested walltime for cpu exceeded 72 hours. Reducing to max. See alphafold-cpu.job: #SBATCH--time","warning")
            return "72:00:00"
        else:
            return f""+time+":00"

def retrieve_gpuwalltime(gputime=""):
    if gputime == "":
        return "24:00:00"
    else:
        times=gputime.split(':')
        total_hours = (int(times[0])+int(times[1])/60)
        if total_hours > 48:
            drona_add_message("Requested walltime for gpu exceeded 48 hours. Reducing to max. See alphafold-gpu.job: #SBATCH --time","warning")
            return "48:00:00"
        else:
            return f""+time+":00"

def process_alpahfold2(version,dbpreset,proteinfasta,modelpreset,datadir,pickle,outputdir):
    if version != "alphafold2":
        return ""
    retrieve_modelpreset(modelpreset)
    retrieve_dbpreset(dbpreset)
    set_modules_and_pickle(pickle)
    retrieve_fasta(proteinfasta)
    retrieve_datadir_vars(datadir)
    return f""



def process_alpahfold3(version,jsoninput,modelinput,recycles,location):
    if version != "alphafold3":
        return ""

    drona_add_message("Drona configures the memory to 64GB. For longer sequences, > 1000 residues, you may need to increase the memory to 128GB. To do so, update the line  #SBATCH --mem=64G  in the alphafold3-cpu.job tab.","note")
    # process alphafold3 params
    drona_add_additional_file("alphafold3-cpu.job","alphafold3-cpu.job")
    drona_add_additional_file("alphafold3-gpu.job","alphafold3-gpu.job")
    drona_add_additional_file("alphafold3-input.json","alphafold3-input.json")

    drona_add_mapping("RECYCLES",str(recycles))
    drona_add_mapping("JSONDIR",location);
    drona_add_mapping("JSONFILE","alphafold3-input.json")
    drona_add_mapping("runcommand",driver_alphafold3)

    if jsoninput == None or jsoninput == "":
        drona_add_message("No input JSON specified. Either provide one or create in preview window","warning")
        drona_add_mapping("JSONINPUT","")
    else:
        content=subprocess.check_output(["cat",jsoninput], universal_newlines=True)
        drona_add_mapping("JSONINPUT",content)
    if modelinput == None or modelinput =="":
        drona_add_message("ERROR: no model provided, your job will not run","error")
        drona_add_mapping("ALPHAFOLD3MODEL","")
    else:
        drona_add_mapping("ALPHAFOLD3MODEL",modelinput)
    return ""


def retrieve_timestamp():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}"
