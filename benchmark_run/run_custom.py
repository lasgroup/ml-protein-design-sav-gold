import os
from stpy.helpers.helper import  cartesian

final_dir = "params/"
d = {
    "kernel": ['full_covariance_se'],
    "topk": [50],
    "feature_selector" : ['lasso'],
    "aminoacid": [1],
    "rosetta": [1],
    "geometric": [1],
    "model": ["ARD"],
	"final_dir": [final_dir],
	"restarts" : [1],
	"maxiters" : [100],
	"cores": [2],
	"cores_split":[5],
	"splits": [5],
	"njobs": [2],
	"special_identifier": ["full_covariance_se"]
}
string_type = [1,0,1,0,0,0,1,1,1,0,0,0,0,0,1]

script_name = "pythonconda run_benchmark.py "
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
