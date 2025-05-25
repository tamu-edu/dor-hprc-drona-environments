import os
import subprocess


def retrieve_cluster():
     result = subprocess.run( ['bash', '-c', 'clustername'],capture_output=True,text=True,check=False)
     return result.stdout.strip()

def retrieve_main(mainscript=""):
    if (mainscript != ""  ):
        return  os.path.basename(mainscript)
    else:
        return ""

def  retrieve_matlabpath(mainscript,matlabroot):
    path_dir = os.path.dirname(mainscript)
    if (matlabroot != ""  ):
        path_dir = path_dir + ":" + matlabroot
    if path_dir != "":
        path_dir="export MATLABPATH="+path_dir+":$MATLABPATH"
    return path_dir	
    
def allow_matlab():
    # On ACES, matlab is group protected. We will check if this Matlab env is on ACES and if so
    # check if the user is in the matlab group

    #check if user is in the MATLAB group on ACES
    cluster = retrieve_cluster();
    if cluster != "aces":
        return True
    else:
        matlab_group = "matlab"
        user_name = os.getenv("USER")
    
        #fetch group information for the matlab group
        try:
            group_info = subprocess.check_output(f"grep {matlab_group} /etc/group", shell=True).decode("utf-8")
        except subprocess.CalledProcessError:
            raise ValueError(f"Group '{matlab_group}' not found in /etc/group")
    
        #output is usually in the format: group_name:x:group_id:user1,user2,...
        group_members = group_info.split(":")[-1].strip().split(",")
    
        #check if the current user is in the MATLAB group
        if user_name in group_members:
            return True
        else:
            return False
    
def retrieve_matlabopts(job_name,workers,threads,walltime,memory,gpu):
    if not allow_matlab():
        drona_add_warning("Matlab is restricted software on ACES. If you are associated with an educational organization, please Contact HPRC to be added to the access list.")
    
    options_string=""
    additional=""
    if gpu != "":
        additional="--partition=gpu "+ gpu + " "
    if walltime != "":
        options_string="-t "+walltime+ " " +options_string
    if workers != "" and workers != "0":
        if int(workers) > 96:
            drona_add_warning("Num workers reduced from " + workers + " to max of 96")
            workers="96"
        options_string="-w " + workers + " " + options_string
    if threads != "" :
        threads=str(min(int(threads),96))
        if  (int(workers)>0 and  (int(workers)*int(threads)) > 96):
            drona_add_warning("Num threads adjusted to fit on a single node")
            threadnum = 96 // min(96,int(workers))
            threads = str(threadnum)
        options_string="-s " + threads + " " + options_string
    if memory != "":
        memory=memory[:-2]
        options_string="-m " + memory + " " + options_string
    if additional != "":
        options_string="-x '" + additional + "' " + options_string
    
    return f""+options_string
