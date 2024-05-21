import torch
import pandas as pd
import numpy as np

from mutedpy.experiments.streptavidin.streptavidin_loader import load_first_round, load_full_emb
from mutedpy.experiments.streptavidin.active_learning.compare_different_models import load_model

def predict_informative(n_informative,
				  model_loc,
				  additional_data,
				  output_filename ='informative-output.csv'  ,
				  steps_by = 200000
				  ):

	# load first round_data
	x, y, dts = load_first_round()

	# load the model
	model, embed = load_model(model_loc, 'fitting')

	# load_phi_test
	xtest, phitest = load_full_emb("AA")

	# add additional_data
	(phi_new,y_new) = additional_data
	phi_old = model.x
	y_old = model.y

	# add together
	phi = torch.vstack((phi_old,phi_new))
	y = torch.vstack((y_old,y_new))

	# fit using all data
	print ("Adding data together")
	model.fit_gp(phi, y)

	print ("Predicting informative, subsample", steps_by)
	most_informative_indices = []
	for j in range(n_informative):
		most_informative_std = 0.
		most_informative_index = 0.

		print("Selecting", j, "/", n_informative)

		sample = np.random.choice(np.arange(0, phitest.size()[0]), steps_by)
		phi_sel = phitest[sample, :]
		_, std = model.mean_std(phi_sel, reuse=True)

		most_informative_std = float(torch.max(std.view(-1)))
		most_informative_index = sample[int(torch.argmax(std.view(-1)))]

		print("selected:", most_informative_index, most_informative_std)

		phinew = phitest[most_informative_index, :].view(1, -1)
		model.add_data_point(phinew, torch.Tensor([[0.]]).double())

		most_informative_indices.append(int(most_informative_index))

		info = xtest[most_informative_indices, :]
		#info.to_csv(output_filename)

if __name__ == "__main__":
	pass