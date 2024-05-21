import os
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.test_functions.protein_benchmark import ProteinOperator

from mutedpy.utils.loaders.loader_basel import BaselLoader
from mutedpy.utils.sequences.sequence_utils import drop_neural_mutations, order_mutations, create_neural_mutations
from mutedpy.protein_learning.gaussian_process.regression_ards_geometry import LassoFeatureSelectorARDGeometric
from mutedpy.protein_learning.gaussian_process.regression_ards import ARDModelLearner
from mutedpy.utils.sequences.sequence_utils import generate_random_mutations,generate_all_combination,add_variant_column,hamming_distance
Op = ProteinOperator()
from dppy.finite_dpps import FiniteDPP


filename = "../../../data/streptavidin/5sites.xls"
loader = BaselLoader(filename)
dts = loader.load()

filename = "../../../data/streptavidin/2sites.xls"
loader = BaselLoader(filename)
total_dts = loader.load(parent = 'SK', positions = [112,121])
total_dts = loader.add_mutations('T111T+N118N+A119A',total_dts)

total_dts = pd.concat([dts, total_dts], ignore_index=True, sort=False)
total_dts = create_neural_mutations(total_dts)
total_dts['LogFitness'] = np.log10(total_dts['Fitness'])

x = torch.from_numpy(Op.translate_mutation_series(total_dts['variant']))
y = torch.from_numpy(total_dts['LogFitness'].values).view(-1, 1)

#############
#############

n = 720

# hopefully high hits
n_optimisic = 72

# should be good hits
n_safe = 72

n_balanced = 72

# informative hits
n_informative = 720-72*3

# predictions for the

pars = {'data_folder':'../data/','features_G': False,'features_V': False, 'kernel':'ard_matern','topk':20, 'restarts':5,"maxiter":500}

model = LassoFeatureSelectorARDGeometric(**pars)
model.preload()
embed = lambda x: torch.hstack([e.embed(x) for e in model.list_of_embeddings])[:, feature_mask]

d = pickle.load(open('final_model_params.p', "rb"))
k = d['kernel_object']
phi =d['x']
yy = d['y']
s = d['noise_std']
feature_mask = d['feature_mask']
GP = GaussianProcess(kernel=k, s=s)
GP.fit_gp(phi, yy)
steps_by = 200000

#N = 10000
#new_mutants = generate_random_mutations(N, [111, 112, 118, 119, 121], ['T', 'S', 'N', 'A', 'K'], prior_muts=total_dts['Mutation'])
new_mutants = generate_all_combination([111,112,118,119,121], ['T','S','N','A','K'])
new_mutants = pd.DataFrame(data = new_mutants,columns=['Mutation'])
new_mutants = add_variant_column(new_mutants)
xtest = torch.from_numpy(Op.translate_mutation_series(new_mutants['variant']))

############################
### Save predictions  ######
############################
phitest = embed(xtest)
new_mutants['mean'] = 0.
new_mutants['std'] = 0.
new_mutants['lcb'] = 0.
new_mutants['ucb'] = 0.

for i in np.arange(0,phitest.size()[0]//steps_by,1):
	print (i,"/",phitest.size()[0]//steps_by)
	mu, std = GP.mean_std(phitest[i*steps_by:(i+1)*steps_by], reuse = True)
	new_mutants['mean'][i*steps_by:(i+1)*steps_by] = mu.view(-1).detach().numpy()
	new_mutants['std'][i * steps_by:(i + 1) * steps_by] = std.view(-1).detach().numpy()
	new_mutants['ucb'][i * steps_by:(i + 1) * steps_by] = mu.view(-1).detach().numpy() + 2*std.view(-1).detach().numpy()
	new_mutants['lcb'][i * steps_by:(i + 1) * steps_by] = mu.view(-1).detach().numpy() - 2*std.view(-1).detach().numpy()

ham = lambda x: hamming_distance(x, total_dts)

rng = np.random.RandomState(1)

#########################
###### OPTIMISTIC    ####
#########################
optimistic_mutants = new_mutants.sort_values(by=['ucb'], ascending=False)[0:500]
optimistic_mutants['Hamming'] = optimistic_mutants['variant'].apply(ham)
optimistic_mutants = optimistic_mutants[optimistic_mutants['Hamming']>0]

x_optimistic = embed(torch.from_numpy(Op.translate_mutation_series(optimistic_mutants['variant'])))
optimistic_mean, K_opt = GP.mean_std(x_optimistic,full=True)

out = optimistic_mutants.iloc[0:n_optimisic,:]
out.to_html("optimistic-best.html")
out.to_csv('optimistic-best.csv')


# select a diverse subset
DPP = FiniteDPP('likelihood', **{'L': K_opt.detach().numpy()})
sample_optimistic = DPP.sample_exact_k_dpp(size=n_optimisic, random_state=rng)
print (optimistic_mutants.iloc[sample_optimistic,:])
x_optimistic = x_optimistic[sample_optimistic,:]
out = optimistic_mutants.iloc[sample_optimistic,:]
out.to_html("optimistic-diverse.html")
out.to_csv('optimistic-diverse.csv')

## SAFE BETS

safe_bets = new_mutants.sort_values(by=['lcb'], ascending=False)[0:500]
safe_bets['Hamming'] = safe_bets['variant'].apply(ham)
safe_bets = safe_bets[safe_bets['Hamming']>0]

x_safe = embed(torch.from_numpy(Op.translate_mutation_series(safe_bets['variant'])))
safe_bets_mean, K_safe = GP.mean_std(x_safe,full=True)

# select a diverse subset
DPP = FiniteDPP('likelihood', **{'L': K_safe.detach().numpy()})
sample_safe = DPP.sample_exact_k_dpp(size=n_safe, random_state=rng)
print (safe_bets.iloc[sample_safe,:])
x_safe = x_safe[sample_safe,:]
out = safe_bets.iloc[sample_safe,:]
out.to_html("safe.html")
out.to_csv('safe.csv')


## BALANCED
balanced = new_mutants.sort_values(by=['mean'], ascending=False)[0:500]
balanced['Hamming'] = balanced['variant'].apply(ham)
balanced = balanced[balanced['Hamming']>0]

x_balanced= embed(torch.from_numpy(Op.translate_mutation_series(balanced['variant'])))
balanced_mean, K_balanced = GP.mean_std(x_safe,full=True)

# select a diverse subset
DPP = FiniteDPP('likelihood', **{'L': K_balanced.detach().numpy()})
sample_balanced = DPP.sample_exact_k_dpp(size=n_balanced, random_state=rng)
print (balanced.iloc[sample_balanced,:])
x_balanced = x_balanced[sample_balanced,:]

out = balanced.iloc[sample_balanced,:]
out.to_html("balanced.html")
out.to_csv('balanced.csv')


## add these to posterior
print (phi.size(), x_optimistic.size())
phi = torch.vstack((phi,x_optimistic))
yy =  torch.vstack((yy,optimistic_mean[sample_optimistic,:]))
phi = torch.vstack((phi,x_balanced))
yy =  torch.vstack((yy,balanced_mean[sample_balanced,:]))
phi = torch.vstack((phi,x_safe))
yy =  torch.vstack((yy,safe_bets_mean[sample_safe,:]))

GP.fit_gp(phi,yy)

print ("Datapoint added.")
#
# ## MOST INFORMATIVE
# most_informative_indices = []
# for j in range(n_informative):
# 	most_informative_std = 0.
# 	most_informative_index = 0.
#
# 	print ("Selecting", j,"/",n_informative)
#
# 	for i in np.arange(0, phitest.size()[0] // steps_by, 1):
#
# 		_, std = GP.mean_std(phitest[i * steps_by:(i + 1) * steps_by], reuse = True)
#
#
# 		new_std = torch.max(std.view(-1))
# 		new_index = torch.argmax(std.view(-1)) + i*steps_by
# 		#print("\t", new_std)
# 		if new_std > most_informative_std:
# 			most_informative_index = int(new_index)
# 			most_informative_std = float(new_std)
# 	print ("selected:",most_informative_index, most_informative_std)
# 	phinew = phitest[most_informative_index,:].view(1,-1)
# 	GP.add_data_point(phinew,torch.Tensor([[0.]]).double())
#
# 	most_informative_indices.append(int(most_informative_index))
#
#
# 	info = new_mutants.loc[most_informative_indices,:]
# 	info['Hamming'] = info['variant'].apply(ham)
# 	info.to_html("informative.html")
# 	info.to_csv('informative.csv')

## MOST INFORMATIVE SUBSET
most_informative_indices = []
for j in range(n_informative):
	most_informative_std = 0.
	most_informative_index = 0.

	print ("Selecting", j,"/",n_informative)

	#for i in np.arange(0, phitest.size()[0] // steps_by, 1):
	sample = np.random.choice(np.arange(0,phitest.size()[0]), steps_by)
	phi_sel = phitest[sample,:]
	_, std = GP.mean_std(phi_sel, reuse = True)

	most_informative_std = float(torch.max(std.view(-1)))
	most_informative_index = sample[int(torch.argmax(std.view(-1)))]
	print ("selected:",most_informative_index, most_informative_std)
	phinew = phitest[most_informative_index,:].view(1,-1)
	GP.add_data_point(phinew,torch.Tensor([[0.]]).double())

	most_informative_indices.append(int(most_informative_index))


	info = new_mutants.loc[most_informative_indices,:]
	info['Hamming'] = info['variant'].apply(ham)
	info.to_html("informative.html")
	info.to_csv('informative.csv')
