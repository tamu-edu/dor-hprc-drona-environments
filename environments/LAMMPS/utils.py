import argparse
import json
import sys
import os


def main(tool, input_data_str, verbose=False):
    # Print the values of the arguments.
    if verbose:
        print(f"Tool: {tool}")
        print(f"Input data: {input_data_str}")
        print("=======")

    try:
        # Attempt to parse the input string as JSON.
        input_data = json.loads(input_data_str)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input.  Details: {e}")
        sys.exit(1)

    if "value" in input_data.keys():
        input_data=input_data["value"]
    
    if tool == 'summary-html':
        if 'summary-html' in input_data.keys():
            summary_value = input_data['summary-html']
            return summary_value
        else:
            print("Error: 'summary' field not found for the selected acceleration strategy.")
            sys.exit(1)
    if tool == 'summary-txt':
        if 'summary-txt' in input_data.keys():
            summary_value = input_data['summary-txt']
            return summary_value
        else:
            print("Error: 'summary' field not found for the selected acceleration strategy.")
            sys.exit(1)
    if tool == 'data':
        if 'data' in input_data.keys():
            data_value = input_data['data']
            return data_value
        else:
            print("Error: 'data' field not found for the selected acceleration strategy.")
            sys.exit(1)
    else:
        print(f"Error: Tool '{tool}' not recognized.")
        sys.exit(1)
    
def get_selected_lmp_cmd(json_input):
    if not json_input:
        
        return "lmp command: Error (No acceleration strategy detected)"
    data=main("data", json_input)
    x=2 # to-do: argument
    mpi_exe="mpirun"
    lmp_exe="lmp"
    prefix=""
    if "mpi_exe" in data.keys() and data["mpi_exe"]:
        mpi_exe=data["mpi_exe"]
    if "lmp_exe" in data.keys() and data["lmp_exe"]:
        lmp_exe=data["lmp_exe"]
    if "prefix" in data.keys() and data["prefix"]:
        prefix=data["prefix"]+" "
    if "framework" in data.keys() and data["framework"]:
        if data["framework"] == "kokkos":
            mpi_cmd=""
            if x>1:
                mpi_cmd=f"{mpi_exe} -np {x} "
            return f"{prefix}{mpi_cmd}{lmp_exe} -k on g {x} -sf kk -pk kokkos gpu/aware on -in $INFILE"
        elif data["framework"] == "gpu":
            mpi_cmd=""
            if x>1:
                mpi_cmd=f"{mpi_exe} -np {x} "
            return f"{prefix}{mpi_cmd}{lmp_exe} -sf gpu -pk gpu {x} -in $INFILE"
        elif data["framework"] == "openmp":
            return f"{prefix}{lmp_exe} -sf omp -pk omp {x} -in $INFILE"
    else:
        return f"{prefix}{lmp_exe} -in $INFILE # (no acceleration framework detected)"

def get_selected_setup(json_input):
    if not json_input:
        return ": Error (No acceleration strategy detected)"
    data=main("data", json_input)
    if "setup" in data.keys() and data["setup"]:
        return data["setup"]
    elif "module" in data.keys() and data["module"]:
        return "module load LAMMPS/"+data["module"]
        return "module list"
    else:
        return "# (no software module detected)"

def get_selected_summary(json_input):
    if not json_input:
        return "module: Error (No acceleration strategy detected)"
    return main("summary-txt", json_input)

def debug_args(*args):
    if args:
        n=len(args)
    else:
        n=0
    return f"# debug print {n} args:\n# "+"\n# ".join(["arg "+str(i)+": "+str(arg) for i, arg in enumerate(args)])

def set_slurm_opts(json_input):
    """
    Sets variables to be used like this:
    #SBATCH --job-name=[JOBNAME]
    #SBATCH --time=[TIME] --mem=[MEM]
    #SBATCH --ntasks=[TASKS] --nodes=[NODES] --cpus-per-task=[CPUS]
    #SBATCH --output=out.%j --error=error.%j
    [EXTRA]
    """
    if not json_input:
        drona_add_message("(No acceleration strategy detected. Your LAMMPS job will not run","error")
        return ""
    extra=""
    x=2 # to-do: argument
    ntasks=x
    cpu_per_task=1
    
    data=main("data", json_input)
    if "accelerator" in data.keys() and data["accelerator"]:
        if data["accelerator"].lower() == "cpu":
            ntasks=1
            cpu_per_task=x
        elif data["accelerator"].lower() == "pvc":
            ntasks=(x//2)*2
            extra+=f"#SBATCH --partition pvc --gres=gpu:pvc:{x}\n"
            extra+=f"#SBATCH --constraint=xelink{x}\n"
        else:
            ntasks=x
            gpu_t=data["accelerator"]
            extra+=f"#SBATCH --partition gpu --gres=gpu:{gpu_t}:{x}\n"
    
    drona_add_mapping("JOBNAME","LAMMPS")
    drona_add_mapping("TIME", "120") # to-do: argument
    drona_add_mapping("NODES","1")
    drona_add_mapping("CPUS",str(cpu_per_task))
    drona_add_mapping("TASKS",str(ntasks))
    drona_add_mapping("MEM",str(40)+"G") # to-do: argument
   
    drona_add_message("Default values used for walltime and memory. You are welcome to adjust these values if needed. Do not change the accelerator settings","note")
    drona_add_message("You can add additional LAMMPS options to the LAMMPS command line and/or adjust the working directory for read/dumps if needed.","note")
    return extra

def compute_workdir(path_to_file):
    return os.path.dirname(path_to_file)

def generate_full_report(filename):
    drona_add_additional_file("report.txt", preview_name="Acceleration report")
    drona_add_message("Check out the Acceleration report for information about the input file analysis.","note")
    from subprocess import run
    try:
        data = run(
            [
                "/usr/bin/bash",
                "-c",
                f"""
                source /sw/hprc/sw/lammps-validator/env.sh && \
                /sw/hprc/sw/lammps-validator/main.py \
                --format None --verbose Info --file {filename}
                """
            ], capture_output=True, text=True)
    except Exception as e:
        return "Error, unable to generate report because:\n{e}"
    if data.stderr:
        return "stderr:\n"+data.stderr+"\nstdout:\n"+data.stdout
    else:
        return data.stdout

if __name__ == "__main__":
    #  This ensures that the main() function is called when the script is executed
    #  from the command line.  It is a common practice in Python scripting.
    """
    This script demonstrates how to use argparse to create a command-line interface (CLI)
    with required arguments: --tool <string> and --input <string>.
    The input is expected to be a JSON string.
    """
    # Create an ArgumentParser object.
    # The description argument provides a brief overview of what the script does.
    parser = argparse.ArgumentParser(description="Extract specific fields from the input.  Input is expected to be a JSON string.")

    parser.add_argument('--tool',
                        dest='tool_name',
                        type=str,
                        required=True,
                        help='The name of the tool to use.')

    parser.add_argument('--input',
                        dest='input_data',  # Changed dest name to reflect JSON data
                        type=str,
                        required=True,
                        help='The input data as a JSON string.')
                        
    parser.add_argument('--verbose',
                        dest='verbose',  # Changed dest name to reflect JSON data
                        type=bool,
                        required=False,
                        help='Extra print statements for debugging.')

    # Parse the command-line arguments.
    args = parser.parse_args()

    tool = args.tool_name
    input_data_str = args.input_data  # Changed variable name
    verbose = args.verbose
    result = main(tool, input_data_str, verbose=verbose)
    print(result)
