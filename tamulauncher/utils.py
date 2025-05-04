import subprocess
import math
import os

def set_norestart(norestart=""):
    if norestart == "":
        return ""
    else:
        return "--norestart"

def retrieve_slurm(nodes,tasks,threads,memory,walltime,account):
    cluster=str(subprocess.run(['/sw/local/bin/clustername' ], stdout=subprocess.PIPE).stdout.decode('utf-8').strip().lower())
    maxcpu=0
    mempercore=0
    if cluster == "grace":
        maxcpu=48
        mempercore=7.5
    elif cluster == "faster":
        maxcpu=64
        mempercore=3.75
    elif cluster ==  "aces":
        maxcpu=96
        mempercore=5
    elif cluster == "launch":
        maxcpu=192
        mempercore=1.88

    drona_add_mapping("NODES",nodes)
    threadnum=int(threads)
    tasknum=0 if tasks=="" else int(tasks)

    # verify the number of threads
    if threadnum > maxcpu:
        drona_add_warning("Warning, #threads cannot exceed max cores on a node. Reducing to max")
        trheadnum = maxcpu

    # verify the number of threads
    if tasknum > maxcpu:
        drona_add_warning("Warning, #command per node cannot exceed max cores on a node. Reducing to max")
        tasknum = maxcpu


    # compute taskls per node
    if tasknum == 0:
        # if task not set, set it to make sure as many cores are being used per node
        tasknum = maxcpu // threadnum
    elif tasknum*threadnum > maxcpu: 
        # if task is set, make sure tasknum*threadnum <= maxcpu
        # if so, adjuist the number of threads
        drona_add_warning("commands per node mulitplied by threads per command, cannot exceeed "+ str(maxcpu)+". Adjusting #threads")
        threadnum= maxcpu // tasknum

    
    # set the timestring
    timestring = "02:00" if walltime == "" else walltime

    memnum=0
    # if memory not set, compute based on mempercore
    mtotal = 0 if memory =="" else int(memory[:-1])

    if mtotal==0:
        memnum = mempercore * threadnum*tasknum
    else:
       memnum = int(totalmemnum // nodes)

    drona_add_mapping("CPUPT",str(threadnum))
    drona_add_mapping("MEM",str(int(memnum))+"G")
    drona_add_mapping("CPUS",str(tasknum*threadnum))
    drona_add_mapping("CPN",str(tasknum))
    drona_add_mapping("TIME",timestring)
    return f""


def retrieve_commandsfile(cfile=""):
    if cfile == "":
        drona_add_warning("ERROR: no commandsfile specified. Your job will fail. Please provide commands file in preview or form)")
    return cfile



def retrieve_loaded_modules(modules=""):
    if modules == "":
        return f""
    else:
        return f" foss/2023b " + modules  





