import math
import os
import subprocess

aces_env_with_star = '''
# load the parabricks and STAR module 

module load Parabricks/4.1.1

module load GCC/8.3.0 STAR/2.7.2b   
'''

aces_env_plain = '''
# load the parabricks module 

module load Parabricks/4.1.1
'''

grace_env_with_star = ''' 
# load modules for GRACE Parabricks and STAR

module load Parabricks/4.0.1

module load GCC/13.3.0 STAR/2.7.2b
'''

grace_env_plain = '''
module load Parabricks/4.0.1
'''

faster_env_plain = '''
module load Parabricks/4.0.1
'''

faster_env_with_star = '''
module load Parabricks/4.0.1

module load GCC/10.2.0 STAR/2.7.2b
'''

def dummy(walltime):
    return f""+walltime

def get_cluster():
    cluster = subprocess.check_output(["/sw/local/bin/clustername"], text=True).strip()
    gpu_set = {}
    if cluster == "aces":
        gpu_set = {"A30", "H100"}

    elif cluster == "grace":
        gpu_set = {"T4", "A100"}

    elif cluster == "faster":
        gpu_set = {"T4", "A10", "A30", "A40"}
    return cluster, gpu_set

def retrieve_tasks_and_other_resources(nodes,tasks,cpus,mem,gpu,numgpu,walltime,account,extra):

   # NODE CONSTANTS
   maxcpunode=96
   maxmemnode=480
   partition = ""

   # tasknum = int(tasks)
   # nodenum  = 0 if nodes == "" else int(nodes)
   
   # Parabricks can only run single node. It's also not compatible with MPI -> only 1 task needed
   tasks = "1"
   tasknum = 1
   nodenum = 1
   
   cpunum = 1 if cpus == "" else int(cpus)
  
   totalmemnum = 0 if mem =="" else int(mem[:-1])
   timestring = "02:00" if walltime == "" else walltime 
  
   # compute the number of hours requested
   times=timestring.split(':') 
   total_hours = (int(times[0])+int(times[1])/60)
   memnum = 0
   partition = ""
   # make sure the number of cpus requested fits on a single node
   if cpunum > maxcpunode:
       drona_add_warning("Requested #cpus_per_task cannot be more than total cores on a node. Reducing #cpus_per_task ")
       cpunum=maxcpunode
   # if nodes is not set, match the number of nodes based on requested tasks and cpus
   if nodenum == 0:
      nodenum = (cpunum*tasknum // maxcpunode) if  (cpunum*tasknum % maxcpunode) == 0 else (cpunum*tasknum // maxcpunode)+1 
   else:
      # check for
      # cpu=1 and tasks < nodes  --> set nodes to match tasks
      # nodes needed to fit cpus*tasks > nodes --> reduce number of cpus     
      if cpunum==1 and tasknum < nodenum:
         drona_add_warning("Requested #tasks < requested #nodes. Need at least one task per node. Adjusting #nodes")
         nodenum=tasknum
      else:
         needed_nodes=(cpunum*tasknum // maxcpunode) if (cpunum*tasknum % maxcpunode) == 0 else (cpunum*tasknum // maxcpunode) +1
         if needed_nodes > nodenum:
            drona_add_warning("#total cores (tasks*cpu) requested needs more nodes than requested. Increasing number of nodes.")
            nodenum=needed_nodes

   memnum = int(totalmemnum // nodenum)
   if memnum == 0: # DEFAULT VALUE
      cpn = (cpunum*tasknum) // nodenum
      memnum = int((maxmemnode/maxcpunode)*cpn)
      drona_add_warning(f"WARNING: Composer setting default total memory to {memnum}G.")
   elif memnum > maxmemnode:
       drona_add_warning("WARNING: Reducing memory to maximum memory per node of " + str(maxmemnode) + "G.")
       memnum = maxmemnode
   # let's check for conflicting requirements


   if nodenum > 8 and (gpu == "h100" or gpu == "a30"):
      drona_add_warning("ERROR: Jobs requesting a GPU Cannot request more than 8 nodes. Your job will not run.")
   elif nodenum > 32 and gpu == "pvc":
      drona_add_warning("ERROR: Jobs requesting a PVC Cannot request more than 32 nodes. Your job will not run.")
   elif nodenum > 64:
      drona_add_warning("ERROR: Limit for cpu jobs is 64 nodes. Your job will not run. Please adjust #nodes.") 

   if gpu != "" and gpu != "none":
       gpunum = 1 if numgpu == "" else numgpu
       ptype = "pvc" if gpu == "pvc" else "gpu"
       if int(gpunum) > 10:
           drona_add_warning("WARNING: max num of gpus is 10, requested " + gpunum + " GPUs. Reducing to max of 10.")
           gpunum="10"
       partition="--partition="+ptype+" --gres=gpu:"+gpu+":"+str(gpunum)
   else:
       partition="--partition=cpu"
   # set the time
   if total_hours == 0:
      drona_add_warning("WARNING: Composer set default time to 2 days.")
      drona_add_mapping("TIME","02:00:00")
   
   else:
      drona_add_mapping("TIME",f""+timestring+":00")


   # combine the extra parameters with partition info and account
   extras = []
   if extra == "$extra":
      extra = ""
   
   if extra != "":
      extras.append(extra)
   
   if partition != "":
      extras.append(partition)

   if account != "":
      account="--account="+account
      extras.append(account)

   if extras:
      drona_add_mapping("EXTRA","#SBATCH " + " ".join(extras))
   else:
      drona_add_mapping("EXTRA", "")

   # we are ready to define all the placeholders now
   drona_add_mapping("NODES",str(nodenum))
   drona_add_mapping("CPUS",str(cpunum))
   drona_add_mapping("MEM",str(memnum)+"G")

   return f""+tasks

def retrieve_loaded_modules(modules=""):
    if modules == "$module_list":
        return ""
    if modules == "":
        return f""
    else:
        return f" foss/2023b " + modules  

# Set to track unique warnings
_unique_warnings = set()

# Wrapper function to add warnings only if they are unique
def add_unique_warning(message):
   if message not in _unique_warnings:
      drona_add_warning(message)
      _unique_warnings.add(message)

def build_parabricks_command(analysisType, ref, fq1, fq2, genomeLibDir, knownSites, outputDir, numgpu, outBam, outVariants, outRecalFile, outPrefix):
   version = ""
   cluster = get_cluster()[0] 
   if cluster == "grace" or cluster == "faster":
       version = "4.0.1"
   else:
       version = "4.1.1"

   if outputDir == "":
      outputDir = "."

   if outBam   == "": 
      outBam   = f"{outputDir}/output.bam"
   else:
      outBam = f"{outputDir}/{outBam}"

   if analysisType == "":
      add_unique_warning("Analaysis Type not specified")

   elif analysisType == "germline":
      if outVariants  == "": 
         outVariants  = f"{outputDir}/output.vcf"
      else:
         outVariants = f"{outputDir}/{outVariants}"
      
      if outRecalFile == "": 
         outRecalFile = f"{outputDir}/output.txt"
      else:
         outRecalFile = f"{outputDir}/{outRecalFile}"

      return f"""singularity exec --nv /sw/hprc/sw/bio/Parabricks/clara-parabricks_{version}-1.sif pbrun germline\\
    --ref {ref}\\
    --in-fq {fq1} {fq2} \\
    --knownSites {knownSites} \\
    --out-bam {outBam} \\
    --out-variants {outVariants} \\
    --out-recal-file {outRecalFile} \\
    --num-gpus {numgpu}"""

   elif analysisType == "rna_fq2bam":
      if outPrefix == "": 
         outPrefix = f"output"
      else:
         outPrefix = f"{outPrefix}"
      return f"""singularity exec --nv /sw/hprc/sw/bio/Parabricks/clara-parabricks_{version}-1.sif pbrun rna_fq2bam\\
      --ref {ref} \\
      --genome-lib-dir {genomeLibDir} \\
      --in-fq {fq1} {fq2} \\
      --output-dir {outputDir} \\
      --out-bam {outBam} \\
      --num-gpus {numgpu} \\
      --out-prefix {outPrefix}"""
   return ""


#Check if a STAR index needs to be generated without adding warnings
def needs_star_index(genomeLibDir):
    if not genomeLibDir:
        return True  # Need to generate index
    
    try:
        if not os.path.isdir(genomeLibDir):
            return True
        files = os.listdir(genomeLibDir)

        required_files = {

            "Genome", "SA", "SAindex",

            "chrLength.txt", "chrName.txt", "chrStart.txt", "genomeParameters.txt"

        }
        has_star_index = required_files.issubset(set(files))

        return not has_star_index
    
    except FileNotFoundError:
        return True

def set_up_env(genomeLibDir, analysisType):
    cluster, gpu = get_cluster()
    if analysisType=="rna_fq2bam" and needs_star_index(genomeLibDir):
        if cluster == 'aces':
            return aces_env_with_star
        elif cluster == 'grace':
            return grace_env_with_star
        else:
            return faster_env_with_star
    else:
        if cluster == 'aces':
            return aces_env_plain
        elif cluster == 'grace':
            return grace_env_plain
        else:
            return faster_env_plain

# STAR genome generation (only for rna_fq2bam)
def build_genome_index(analysisType, genomeLibDir, ref, genomeAnnotation, cpus):
    if analysisType != "rna_fq2bam":
        return ""
    
    if not genomeLibDir:
        add_unique_warning("Genome Library Directory not provided, will generate genome index.")
        needs_index = True
    
    else:
        needs_index = needs_star_index(genomeLibDir)
        if needs_index:
            add_unique_warning("STAR index missing or incomplete. Will generate genome index.")

    if not needs_index:
        return ""

    star_command = """STAR --runThreadN {cpus} \\
    --runMode genomeGenerate \\
    --genomeDir {genomeLibDir} \\
    --genomeFastaFiles {ref}{gtf_line} 
    {sjdbOverhang}
    """.format(
        cpus=cpus,
        genomeLibDir=genomeLibDir,
        ref=ref,
        sjdbOverhang=f"--sjdbOverhang 99" if genomeAnnotation else "",
        gtf_line=f" \\\n   --sjdbGTFfile {genomeAnnotation}" if genomeAnnotation else ""
        )
    
    return star_command
