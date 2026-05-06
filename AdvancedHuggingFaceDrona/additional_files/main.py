# main.py
import os
import re
import subprocess
import time

def main():
    """Submits the SLURM job and streams the logs robustly."""
    print("="*60)
    print("      Drona SLURM Job Launcher      ".center(60))
    print("="*60)
    print(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    slurm_script = "run_operation_slurm.sh"
    print(f"\nSubmitting SLURM job using 'sbatch {slurm_script}'...")
    try:
        result = subprocess.run(['sbatch', slurm_script], capture_output=True, text=True, check=True)
        match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if not match:
            print("--> Error: Could not parse Job ID from sbatch output.")
            return
        job_id = match.group(1)
        print(f"--> SLURM job submitted successfully. Job ID: {job_id}")

        log_file = f"out.{job_id}"
        print(f"--> Waiting for output file: {log_file}")
        while not os.path.exists(log_file):
            time.sleep(2)
        print("--> Output file found. Streaming logs in real-time...\n" + "-"*60)

        # ====================================================================
        # THE FIX: Open the file in binary mode ('rb') to prevent decode errors.
        # ====================================================================
        with open(log_file, 'rb') as f:
            while True:
                job_status_result = subprocess.run(['squeue', '-j', job_id], capture_output=True, text=True)
                is_running = job_id in job_status_result.stdout

                # Read raw bytes from the file
                line_bytes = f.readline()

                if not line_bytes:
                    if not is_running:
                        break # Exit if job is done and no more output
                    time.sleep(1)
                    continue
                else:
                    # Decode the bytes into a string, replacing any broken characters
                    # This makes the stream robust to progress bars and special characters.
                    line_str = line_bytes.decode('utf-8', errors='replace')
                    # Use end='' because the line already has a newline
                    print(line_str, end='')
        # ====================================================================

        print("\n" + "-" * 60 + "\n--> Job has completed. Log streaming finished.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        print("\n" + "="*60)
        print(f"Job launcher finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

if __name__ == "__main__":
    main()