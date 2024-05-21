import os
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt

from mutedpy.protein_learning.embeddings.amino_acid_embedding import AminoAcidEmbedding
from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.test_functions.protein_benchmark import ProteinOperator
from mutedpy.utils.sequences.sequence_utils import generate_random_mutations,add_variant_column


def load_model(params,log_name = 'none', model_params_return=False, change_database = None, vintage = False):

	model_params = pickle.load(open(params, "rb"))
	k = model_params['kernel_object']
	phi = model_params['x']
	yy = model_params['y']

	s = model_params['noise_std']
	if vintage:
		print (model_params.keys())
	else:
		feature_loader = model_params['feature_loader']
	feature_mask = model_params['feature_mask']
	GP = GaussianProcess(kernel=k, s=s)
	print("Fitting", log_name)
	print(k.params)
	if 'Sigma' in model_params.keys():
		GP.fit_gp(phi, yy, Sigma = model_params['Sigma'])
	else:
		GP.fit_gp(phi, yy)
	if change_database is not None:
		feature_loader.feature_loaders[0].Embedding.project = change_database
	if vintage:
		path_dir = os.path.dirname(__file__)
		feature_loader = AminoAcidEmbedding(data=path_dir + "/../data/amino-acid-features.csv",
											 projection=path_dir + "/../data/embedding-dim5-demean-norm.pt",
											 proto_names=5)
		embed = lambda x: feature_loader.embed(x)[:,feature_mask]
	else:
		feature_loader.connect()
		embed = lambda x: feature_loader.embed(x)[:, feature_mask]
	if not model_params_return:
		return GP,embed
	else:
		return GP, embed, model_params

def load_model_and_mean_std(params,log_name,xtest, change_database = None, vintage = False):
	GP,embed = load_model(params,log_name, change_database = change_database, vintage = vintage)
	phitest = embed(xtest)
	ytest, std = GP.mean_std(phitest)
	return ytest, std, GP

def generate_new_mutant_file(load_function,N,comparison_mutant_file, parent ,positions):
	x, y, total_dts = load_function()
	new_mutants = generate_random_mutations(N, positions, parent,
											prior_muts=total_dts['Mutation'])
	new_mutants = pd.DataFrame(data=new_mutants, columns=['Mutation'])
	new_mutants = add_variant_column(new_mutants)
	new_mutants.to_csv(comparison_mutant_file)
	return new_mutants


def compare_from_params(models, embeds, comparison_mutant_file, N = 10000):
	Op = ProteinOperator()
	print ("Generating mutant file.")
	new_mutants = generate_random_mutations(N,[111, 112, 118, 119, 121],['T', 'S', 'N', 'A', 'K'])
	new_mutants = pd.DataFrame(data=new_mutants, columns=['Mutation'])
	new_mutants = add_variant_column(new_mutants)
	xtest = torch.from_numpy(Op.translate_mutation_series(new_mutants['variant']))
	res = []
	for embed, model in zip(embeds, models):
		phi = embed(xtest)
		mean, std = model.mean_std(phi)
		res.append(mean)
	R2_matrix = np.corrcoef(np.concatenate(res, axis = 1), rowvar=False)
	plt.imshow(R2_matrix)
	plt.colorbar()
	plt.show()


def compare(models, comparison_name, comparison_mutant_file, results_folder, splits = 10, N = 10000, load_function = None, finals = None):

	Op = ProteinOperator()
	if os.path.exists(comparison_mutant_file):
		print ("File exists.")
		new_mutants = pd.read_csv(comparison_mutant_file)
		xtest = torch.from_numpy(Op.translate_mutation_series(new_mutants['variant']))

	else:
		print ("Generating mutant file.")
		new_mutants = generate_random_mutations(load_function,N,comparison_mutant_file,['T', 'S', 'N', 'A', 'K'],[111, 112, 118, 119, 121])
		xtest = torch.from_numpy(Op.translate_mutation_series(new_mutants['variant']))

	res = []
	names = []
	for model_name in models:
		for split in range(splits):
			names.append(model_name + str(split))
			params = results_folder + model_name + "/model_params_" + str(split) + ".p"
			ytest, std, GP = load_model_and_mean_std(params,str(split),xtest)
			res.append(ytest.detach().numpy())

	if finals is not None:
		for model_loc in finals:
			params = model_loc
			names.append("final")
			ytest, std, GP = load_model_and_mean_std(params, str(split), xtest)
			res.append(ytest.detach().numpy())

	R2_matrix = np.corrcoef(np.concatenate(res, axis = 1), rowvar=False)

	plt.imshow(R2_matrix)
	plt.colorbar()
	if finals is not None:
		plt.xticks([i for i in range(len(models)*splits+len(finals))],names,rotation=90)
		plt.yticks([i for i in range(len(models)*splits+len(finals))],names)
	else:
		plt.xticks([i for i in range(len(models) * splits)], names, rotation=90)
		plt.yticks([i for i in range(len(models) * splits)], names)

	plt.savefig(comparison_name, dpi = 200)
	plt.show()

if __name__ == "__main__":
	models = ["new_ardlasso_0_0_1_100",
			"new_ardlasso_0_0_1_20",
			"new_ardlasso_0_0_1_30",
		"new_ardlasso_0_0_1_50",]

	results_folder = "../../results_strep/"
	splits = 10
	comparison_name = "AA_model/plots/amino-acid-comparison.png"

	comparison_mutants_file = 'lists/random_mutants.csv'
	compare(models, comparison_name, comparison_mutants_file,results_folder, splits = splits)


