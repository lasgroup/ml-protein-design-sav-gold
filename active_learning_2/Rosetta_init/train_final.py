import os
from stpy.helpers.helper import  cartesian

final_dir = "params/"
d = {
    "kernel": ['ard_matern'],
    "topk": [100],
    "feature_selector" : ['lasso'],
    "aminoacid": [1],
    "rosetta": [1],
    "geometric": [1],
    "model": ["ARD"],
	"target": ["final_model"],
	"final_dir": [final_dir],
	"restarts" : [3],
	"maxiters" : [100],
	"special_identifier": ["final"],
    "data":["total"],
	"split_loc":["../splits/random_splits_total.p"]
}
string_type = [1,0,1,0,0,0,1,1,1,0,0,1,1,1,1]

script_name = "pythonconda ../../benchmark_run/run_benchmark.py "
v = [d[key][0] for key in d.keys()]
command = script_name
names = list(d.keys())
for index, arg in enumerate(v):
	if string_type[index]:
		command += "--"+names[index] + "='"+str(arg)+"' "
	else:
		command += "--"+names[index] + "=" + str(arg) + " "
print (command)
os.system(command)
