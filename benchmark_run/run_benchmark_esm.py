import pymongo
import time
import os
import numpy as np

from stpy.helpers.helper import  cartesian
d = {
    "kernel": ['ard_matern'],
    "topk": [10,20,40,80,100,512,250,1280],
    "feature_selector" : ["lasso","none"],
    "aminoacid": [0,1],
    "rosetta": [0],
    "geometric": [0],
    "esm": [1],
    "project_geo": ["streptavidin-gold-def"],
    "transformation" : ['log'],
    "model": ["linear","ARD"],
    "special_identifier":["test_esm_March2024_"],
    "maxiters": [200],
    "restarts": [3],
    "njobs":[5],
    "cores_split":[3],
    "splits":[15],
    "model_selection_data":["train"],
    "results_folder":["results_strep_esm"],
    "additive": [0, 1]
}

string_type = [1,0,1,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0]
script_name = "cd ../ && python run_benchmark.py "
dtype = np.asarray(["streptavidin-gold","streptavidin-gold-def","streptavidin-gold-opt"]).dtype
v = cartesian([a for a in d.values()], dtype = dtype)

def preamble(index):
    return """#!/bin/bash 
#SBATCH -n 15
#SBATCH --time=03:59:00
#SBATCH --mem-per-cpu=1000
#SBATCH --tmp=1000                        # per node!!
#SBATCH --job-name=be"""+str(index)+"""
#SBATCH --output=/cluster/project/krause/mmutny/"""+str(index)+"""-benchmark_run.stdout
#SBATCH --error=/cluster/project/krause/mmutny/"""+str(index)+"""-benchmark_run.stderr"""


for counter, val in enumerate(v):
    command = script_name
    names = list(d.keys())
    for index, arg in enumerate(val):
        if string_type[index]:
            command += "--"+names[index] + "='"+arg+"' "
        else:
            command += "--"+names[index] + "=" + arg + " "
    print (command)
    f = open("job_files_esm/job"+str(counter)+".sh","w")
    f.writelines(preamble(counter)+"\n")
    f.writelines(command)
    f.close()


