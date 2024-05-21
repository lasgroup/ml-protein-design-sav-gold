import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from mutedpy.experiments.streptavidin.streptavidin_loader import tobias_colors

"""
This script generates a table with the results from a specific directory for visual comparsion 
"""

bottom = 0.2
directory = "results_strep"
model_dirs = [f.path for f in os.scandir(directory) if f.is_dir()]
del model_dirs[0]
dts = []

for dir in model_dirs:
	try:
		if "all" in dir or "train" in dir:
			d = pd.read_csv(dir + "/rawoutput.csv")
		else:
			d = None
	except:
		#print ("Not found", dir)
		d = None

	if d is not None:
		#print (d)
		name = dir.split("/")[-1]
		if "DEC" in dir:
			#print (name)
			d['name'] = name
			d['selector'] 	= d['name'].apply(lambda x: x.split("_")[3])
			d['fea_size'] 	= d['name'].apply(lambda x: int(x.split("_")[7]))
			d['structures'] = d['name'].apply(lambda x: x.split("_")[8])
			d['data'] 		= d['name'].apply(lambda x: x.split("_")[12])
			def f(x):
				if x.split("_")[4] == "1" and x.split("_")[5] == "1":
					return "Nonlinear / Structure+Energy"
				elif x.split("_")[4] == "1" and x.split("_")[5] == "0":
					return "Nonlinear / Energy"
				elif x.split("_")[4] == "0" and x.split("_")[5] == "1":
					return "Nonlinear / Structure"
				else:
					return "Nonlinear / Chemical"

			d['fea_name'] = d['name'].apply(f)
			#print(d['fea_name'][0], name)
			dts.append(d)
		elif "Oct" in dir and "linear" in dir:
			#print (name)
			d['name'] = name
			d['selector'] 	= d['name'].apply(lambda x: x.split("_")[3])
			d['fea_size'] 	= d['name'].apply(lambda x: int(x.split("_")[7]))
			d['structures'] = d['name'].apply(lambda x: x.split("_")[8])
			d['data'] 		= d['name'].apply(lambda x: x.split("_")[12])
			def f(x):
				if x.split("_")[4] == "1" and x.split("_")[5] == "1":
					return "Linear / Structure+Energy"
				elif x.split("_")[4] == "1" and x.split("_")[5] == "0":
					return "Linear / Energy"
				elif x.split("_")[4] == "0" and x.split("_")[5] == "1":
					return "Linear / Structure"
				else:
					return "Linear / Chemical"
			d['fea_name'] = d['name'].apply(f)
			#print(d['fea_name'][0], name)
			dts.append(d)
		elif "Oct" in dir and "ARD" in dir :
			#print (name)
			d['name'] = name
			d['selector'] = d['name'].apply(lambda x: x.split("_")[3])
			d['fea_size'] = d['name'].apply(lambda x: int(x.split("_")[7]))
			d['structures'] = d['name'].apply(lambda x: x.split("_")[8])
			d['data'] = d['name'].apply(lambda x: x.split("_")[12])


			def f(x):
				if x.split("_")[4] == "1" and x.split("_")[5] == "1":
					return "Additive / Structure+Energy"
				elif x.split("_")[4] == "1" and x.split("_")[5] == "0":
					return "Additive / Energy"
				elif x.split("_")[4] == "0" and x.split("_")[5] == "1":
					return "Additive / Structure"
				else:
					return "Additive / Chemical"


			d['fea_name'] = d['name'].apply(f)
			# print(d['fea_name'][0], name)
			dts.append(d)

		elif "Jan" in dir and "ARD" in dir and "noise" in dir and "bench_noise_Jan" not in dir:
			print (name)
			d['name'] = name
			d['selector'] = d['name'].apply(lambda x: x.split("_")[3])
			d['fea_size'] = d['name'].apply(lambda x: int(x.split("_")[7]))
			d['structures'] = d['name'].apply(lambda x: x.split("_")[8])
			d['data'] = d['name'].apply(lambda x: x.split("_")[12])


			def f(x):
				if int(x.split("_")[14]) == 0:
					if x.split("_")[4] == "1" and x.split("_")[5] == "1":
						return "Nonlinear / Structure+Energy"
					elif x.split("_")[4] == "1" and x.split("_")[5] == "0":
						return "Nonlinear / Energy"
					elif x.split("_")[4] == "0" and x.split("_")[5] == "1":
						return "Nonlinear / Structure"
					else:
						return "Nonlinear / Chemical"
				else:
					if x.split("_")[4] == "1" and x.split("_")[5] == "1":
						return "Additive / Structure+Energy"
					elif x.split("_")[4] == "1" and x.split("_")[5] == "0":
						return "Additive / Energy"
					elif x.split("_")[4] == "0" and x.split("_")[5] == "1":
						return "Additive / Structure"
					else:
						return "Additive / Chemical"

			d['fea_name'] = d['name'].apply(f)
			# print(d['fea_name'][0], name)
			dts.append(d)

		elif "bench_noise_Jan" in dir and "ARD" in dir and "noise" in dir:
			print (name)
			d['name'] = name
			d['selector'] = d['name'].apply(lambda x: x.split("_")[4])
			d['fea_size'] = d['name'].apply(lambda x: int(x.split("_")[8]))
			d['structures'] = d['name'].apply(lambda x: x.split("_")[9])
			d['data'] = d['name'].apply(lambda x: x.split("_")[13])
			d['noise'] = d['name'].apply(lambda x: float(x.split("_")[16][5:]))

			def f(x):
				if int(x.split("_")[15]) == 0:
					if x.split("_")[5] == "1" and x.split("_")[6] == "1":
						return "Nonlinear / Structure+Energy"
					elif x.split("_")[5] == "1" and x.split("_")[6] == "0":
						return "Nonlinear / Energy"
					elif x.split("_")[5] == "0" and x.split("_")[6] == "1":
						return "Nonlinear / Structure"
					else:
						return "Nonlinear / Chemical"
				else:
					if x.split("_")[5] == "1" and x.split("_")[6] == "1":
						return "Additive / Structure+Energy"
					elif x.split("_")[5] == "1" and x.split("_")[6] == "0":
						return "Additive / Energy"
					elif x.split("_")[5] == "0" and x.split("_")[6] == "1":
						return "Additive / Structure"
					else:
						return "Additive / Chemical"

			d['fea_name'] = d['name'].apply(f)
			d['name'] = "bench_noise"
			# print(d['fea_name'][0], name)
			dts.append(d)

		elif "Jan" in dir and "linear" in dir and "noiseNone" in dir:
			d['name'] = name
			d['selector'] 	= d['name'].apply(lambda x: x.split("_")[3])
			d['fea_size'] 	= d['name'].apply(lambda x: int(x.split("_")[7]))
			d['structures'] = d['name'].apply(lambda x: x.split("_")[8])
			d['data'] 		= d['name'].apply(lambda x: x.split("_")[12])
			def f(x):
				if int(x.split("_")[14]) == 0:
					if x.split("_")[4] == "1" and x.split("_")[5] == "1":
						return "Linear / Structure+Energy"
					elif x.split("_")[4] == "1" and x.split("_")[5] == "0":
						return "Linear / Energy"
					elif x.split("_")[4] == "0" and x.split("_")[5] == "1":
						return "Linear / Structure"
					else:
						return "Linear / Chemical"
			d['fea_name'] = d['name'].apply(f)
			#print(d['fea_name'][0], name)
			dts.append(d)


plot_names = ["r2","Pearson","Spearman","hit_rate"]

final_dts = pd.concat(dts)

# filter with feature size
mask_feasize = final_dts['fea_size'] <= 50
final_dts = final_dts[mask_feasize]

# filter with considered models
ml_models = ["Nonlinear / Chemical",
			 "Nonlinear / Structure",
			 "Nonlinear / Structure+Energy",
			 "Linear / Chemical",
			 "Linear / Structure",
			"Linear / Structure+Energy",
			 "Additive / Chemical"]

mask_models = final_dts['fea_name'].isin(ml_models)
final_dts = final_dts[mask_models]

print (final_dts['fea_name'].unique())

# change the names of the collumns
column_names = {
    'r2': '$R^2$ on test set' ,
    'fea_name': 'Model / Descriptors',
    'fea_size': 'Number of Descriptors',
	'hit_rate': 'Hit Rate',
	'pearson':'Pearson',
	'spearman':'Spearman'

}




plot_names = ['$R^2$ on test set',"Pearson","Spearman","Hit Rate"]

palette = {"Nonlinear / Chemical": '#6364AD',
		   "Nonlinear / Structure": '#947AB7',
		   "Nonlinear / Structure+Energy": '#A467AA',
		   "Nonlinear / Energy": "tab:red",
		   "Linear / Chemical": "#6364AD",
		   "Linear / Structure+Energy": "#947AB7",
		   "Linear / Structure": "#A467AA",
		    "Linear / Energy": "tab:red",
		   "Additive / Chemical": "#6E796E",
		   "Additive / Structure+Energy": "#A467AA",
		   "Additive / Structure": "pink",
			"Additive / Energy": "tab:red"
		   }


## data plots
final_dts = final_dts.rename(columns=column_names)
final_dts = final_dts.sort_values(by=['Model / Descriptors'])

legacy_final_dts =final_dts.copy()
final_dts = final_dts[final_dts['name']!="bench_noise"]
final_dts[~final_dts['noise'].isna()]['noise'] = 0.146

print (final_dts)

final_dts1 = final_dts[final_dts['Model / Descriptors']=="Nonlinear / Chemical"]
final_dts1 = final_dts1[final_dts1['structures']=="streptavidin-gold-def"]

print (final_dts1)
for fname in plot_names:
	sns.boxplot(x='Number of Descriptors', y=fname,
				hue=final_dts1[['selector', 'data']].apply(tuple, axis=1), palette="tab20",
				data=final_dts1)
	sns.despine(offset=10, trim=True)
	plt.subplots_adjust(bottom=bottom)
	#plt.legend(ncol=2, loc= "lower center")
	plt.savefig('results_table/' + fname + "-data.pdf")
	plt.clf()


## selector plots

final_dts2 = final_dts[final_dts['data']=='train']
final_dts2 = final_dts2[final_dts2['structures']=='streptavidin-gold-def']
print (final_dts2)

for fname in plot_names:
	sns.boxplot(x='Number of Descriptors', y=fname,
				hue=final_dts2[['selector']].apply(tuple, axis=1), palette="tab10",
				data=final_dts2)
	sns.despine(offset=10, trim=True)
	plt.legend(ncol=2, loc= "lower center")
	plt.savefig('results_table/' + fname + "-selector.pdf")
	plt.subplots_adjust(bottom=bottom)
	plt.clf()


final_dts3 = final_dts[final_dts['selector']=='lasso']
final_dts3 = final_dts3[final_dts3['data']=='train']
final_dts3 = final_dts3[final_dts3['structures']=='streptavidin-gold-def']

## whole plots
final_dts3.to_csv("selection.csv")
for fname in plot_names:
	ax = sns.boxplot(x='Number of Descriptors', y=fname,
				hue='Model / Descriptors', palette=palette,
				data=final_dts3)

	sns.despine(offset=10, trim=True)
	plt.subplots_adjust(bottom=bottom)
	plt.legend(ncol=2, loc= "lower center")

	plt.savefig('results_table/' + fname + "-whole.pdf")
	plt.clf()

	#plt.show()


ml_models_chemical = ["Nonlinear / Chemical",
					 "Linear / Chemical",
					"Additive / Chemical"]

ml_models_linear = ["Linear / Chemical",
					"Linear / Structure",
					"Linear / Structure+Energy"]

ml_models_nonlinear = ["Nonlinear / Chemical",
					 "Nonlinear / Structure",
					 "Nonlinear / Structure+Energy",
					   "Additive / Chemical",
					   "Additive / Structure",
					   "Additive / Structure+Energy"]


mask_models = final_dts3['Model / Descriptors'].isin(ml_models_chemical)
final_dts10 = final_dts3[mask_models]
## make one plot with only chemical
for fname in plot_names:
	ax = sns.boxplot(x='Number of Descriptors', y=fname,
				hue='Model / Descriptors', palette=palette,
				data=final_dts10)

	sns.despine(offset=10, trim=True)
	plt.subplots_adjust(bottom=bottom)
	plt.legend(ncol=2, loc= "lower center")

	plt.savefig('results_table/' + fname + "-whole-chemical.pdf")
	plt.clf()

## make two plots one with chemical features only linear
mask_models = final_dts3['Model / Descriptors'].isin(ml_models_linear)
final_dts10 = final_dts3[mask_models]
sns.set_theme(style="whitegrid")
plt.rcParams['hatch.linewidth'] = 3.0  # Default is 1.0, increase for thicker lines

## make one plot with only chemical
for fname in plot_names:
	ax = sns.boxplot(x='Number of Descriptors', y=fname,
				hue='Model / Descriptors', palette=palette,
				data=final_dts10, fliersize=0)
	for bar in ax.patches:
		bar.set_hatch('/')
		bar.set_edgecolor('white')
	sns.despine(offset=10, trim=True)

	plt.subplots_adjust(bottom=bottom)
	plt.legend(ncol=1, loc= "lower center")

	plt.savefig('results_table/' + fname + "-whole-linear.pdf")
	plt.clf()


## make two plots one with chemical features only non-linear
mask_models = final_dts3['Model / Descriptors'].isin(ml_models_nonlinear)
final_dts10 = final_dts3[mask_models]
## make one plot with only chemical
for fname in plot_names:
	ax = sns.boxplot(x='Number of Descriptors', y=fname,
				hue='Model / Descriptors', palette=palette,
				data=final_dts10)

	sns.despine(offset=10, trim=True)
	plt.subplots_adjust(bottom=bottom)
	plt.legend(ncol=2, loc= "lower center")

	plt.savefig('results_table/' + fname + "-whole-nonlin.pdf")
	plt.clf()







final_dts4 = final_dts[final_dts['selector']=='lasso']
final_dts4 = final_dts4[final_dts4['Model / Descriptors']=='Nonlinear / Chemical']

for fname in plot_names:
	sns.boxplot(x="Number of Descriptors", y=fname,
				hue='structures', palette= "tab20",
				data=final_dts4)
	sns.despine(offset=10, trim=True)
	plt.subplots_adjust(bottom=bottom)
	plt.savefig('results_table/' + fname + ".pdf",bbox_inches='tight')
	plt.clf()
	#plt.show()
print (final_dts)
final_dts = legacy_final_dts[legacy_final_dts['name']=="bench_noise"]
print (final_dts)
final_dts5 = final_dts[final_dts['selector'] == 'lasso']
final_dts5 = final_dts5[final_dts5['Model / Descriptors'] == 'Nonlinear / Chemical']
print (final_dts5['noise'])
print (len(final_dts5))
for fname in plot_names:
	sns.boxplot(x="noise", y=fname,palette=["tab:blue"] * 3 + ['tab:orange'] +["tab:blue"] * 3,	data=final_dts5)
	sns.despine(offset=10, trim=True)
	plt.subplots_adjust(bottom=bottom)
	plt.savefig('results_table/'+fname+'-noise.pdf', bbox_inches='tight')
	plt.clf()
# plt.show()






