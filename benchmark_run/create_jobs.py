import pymongo
import time
import os
import numpy as np

from stpy.helpers.helper import  cartesian
d = {
    "kernel": ['ard_matern'],
    "topk": [20, 30, 50, 60, 80, 100],
    "feature_selector" : ["lasso","elastic", "rf"],
    "aminoacid": [1],
    "rosetta": [0,1],
    "geometric": [0, 1],
    "project_geo": ["streptavidin-gold-def","streptavidin-gold","streptavidin-gold-opt"],
    "transformation" :['log'],
    "model": ["ARD"],
    "special_identifier":["struct_test_MAY2023_"],
    "maxiters": [250],
    "restarts": [3],
    "njobs":[5],
    "cores_split":[3],
    "splits":[15],
    "model_selection_data":["train"]
}

string_type = [1,0,1,0,0,0,1,1,1,1,0,0,0,0,0,1]
script_name = "cd ../ && python run_benchmark.py "
dtype = np.asarray(["streptavidin-gold","streptavidin-gold-def","streptavidin-gold-opt"]).dtype
v = cartesian([a for a in d.values()], dtype = dtype)

def preamble(index):
    return """#!/bin/bash 
#SBATCH -n 15
#SBATCH --time=03:59:00
#SBATCH --mem-per-cpu=1000
#SBATCH --tmp=1000                        # per node!!
#SBATCH --job-name=extra"""+str(index)+"""
#SBATCH --output=/cluster/project/krause/mmutny/"""+str(index)+"""-benchmark_extra.stdout
#SBATCH --error=/cluster/project/krause/mmutny/"""+str(index)+"""-benchmark_extra.stderr"""


for counter, val in enumerate(v):
    command = script_name
    names = list(d.keys())
    for index, arg in enumerate(val):
        if string_type[index]:
            command += "--"+names[index] + "='"+arg+"' "
        else:
            command += "--"+names[index] + "=" + arg + " "
    print (command)
    f = open("job_files_exp/job"+str(counter)+".sh","w")
    f.writelines(preamble(counter)+"\n")
    f.writelines(command)
    f.close()




