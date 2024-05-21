import os
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from stpy.continuous_processes.gauss_procc import GaussianProcess
from mutedpy.utils.protein_operator import ProteinOperator
from mutedpy.utils.loaders.loader_basel import BaselLoader
from mutedpy.utils.sequences.sequence_utils import drop_neural_mutations, order_mutations, create_neural_mutations
from mutedpy.protein_learning.regression.regression_ards_geometry import LassoFeatureSelectorARDGeometric
from mutedpy.protein_learning.regression.regression_ards import ARDModelLearner
from mutedpy.utils.sequences.sequence_utils import generate_random_mutations,generate_all_combination,add_variant_column,hamming_distance
from mutedpy.experiments.streptavidin.active_learning.compare_different_models import load_model_and_mean_std, load_model
### Load_model


def generate_all_pairs(positions,parent):
	Op = ProteinOperator()
	new_mutants = generate_all_combination(positions, parent)
	new_mutants = pd.DataFrame(data=new_mutants, columns=['Mutation'])
	new_mutants = add_variant_column(new_mutants)
	xtest = torch.from_numpy(Op.translate_mutation_series(new_mutants['variant']))
	return xtest, new_mutants

def generate_all_predictions(model_params,positions,parent, steps_by = 32000):
	xtest ,new_mutants= generate_all_pairs(positions,parent)
	GP, embed =  load_model(model_params,"log")
	phitest = embed(xtest)
	return predictions_separated(new_mutants,GP,xtest,phitest,steps_by)

def predictions_separated(new_mutants,GP,xtest, phitest, steps_by):
	new_mutants['mean'] = 0.
	new_mutants['std'] = 0.
	new_mutants['lcb'] = 0.
	new_mutants['ucb'] = 0.
	for i in np.arange(0,xtest.size()[0]//steps_by,1):
		print (i,"/",xtest.size()[0]//steps_by)
		mu, std = GP.mean_std(phitest[i*steps_by:(i+1)*steps_by], reuse = True)
		new_mutants['mean'][i*steps_by:(i+1)*steps_by] = mu.view(-1).detach().numpy()
		new_mutants['std'][i * steps_by:(i + 1) * steps_by] = std.view(-1).detach().numpy()
		new_mutants['ucb'][i * steps_by:(i + 1) * steps_by] = mu.view(-1).detach().numpy() + 2*std.view(-1).detach().numpy()
		new_mutants['lcb'][i * steps_by:(i + 1) * steps_by] = mu.view(-1).detach().numpy() - 2*std.view(-1).detach().numpy()
	return new_mutants

def prediction_step_by_step(GP,phitest,steps_by):
	mu_whole = torch.zeros(size = (phitest.size()[0],1)).double()
	std_whole = torch.zeros(size = (phitest.size()[0],1)).double()
	for i in np.arange(0,phitest.size()[0]//steps_by,1):
		print (i,"/",phitest.size()[0]//steps_by)
		mu, std = GP.mean_std(phitest[i*steps_by:(i+1)*steps_by], reuse = True)
		mu_whole[i*steps_by:(i+1)*steps_by,0] = mu.view(-1).detach()
		std_whole[i * steps_by:(i + 1) * steps_by, 0] = std.view(-1).detach()
	return (mu_whole,std_whole)

def add_embelishments(new_mutants, callable = None):
	if callable is None:
		callable = lambda x: x
	new_mutants = new_mutants.sort_values(by=['mean'], ascending=False)

	new_mutants['Predicted Fitness'] = callable(10**new_mutants['mean'])
	new_mutants['Optimistic Fitness'] = callable(10**new_mutants['ucb'])
	new_mutants['Pessimistic Fitness'] = callable(10**new_mutants['lcb'])
	# new_mutants['Hamming'] = new_mutants['variant'].apply(ham)
	return new_mutants


if __name__ == "__main__":
	positions = [111,112,118,119,121]
	parent = 'TSNAK'

	model_params = "AA_model/params/final_model_params.p"
	new_mutants = generate_all_predictions(model_params,positions,parent)
	new_mutants = add_embelishments(new_mutants)

	new_mutants.to_csv('AA_model/lists/predictions-aa.csv')
	#new_mutants.to_html('AA_model/lists/predictions-aa.html')

	model_params = "Geometric/params/final_model_params.p"
	new_mutants = generate_all_predictions(model_params,positions,parent)
	new_mutants = add_embelishments(new_mutants)

	new_mutants.to_csv('Geometric/lists/predictions-geo.csv')
	#new_mutants.to_html('Geometric/lists/predictions-geo.html')
