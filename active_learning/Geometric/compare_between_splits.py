from mutedpy.experiments.streptavidin.active_learning.compare_different_models import compare
from mutedpy.protein_learning.active_learning.generate_predictions import load_model
if __name__ == "__main__":

	models = ["struct_test_DEC2022_lasso_0_1_1_50_streptavidin-gold_ard_matern_ARD_all"]

	results_folder = "../../results_strep/"
	splits = 10
	comparison_name = "plots/amino-acid-comparison-DEC_2022.png"
	comparison_mutants_file = '../AA_model/lists/random_mutants.csv'

	model, embed, d = load_model("params/final_model_params.p", model_params_return = True)
	model_new, embed_new, d_new = load_model("params/dec_2022_finalfinal_model_params.p", model_params_return=True)

	feature_loader = d['feature_loader']
	feature_loader.feature_loaders[0].Embedding.project = 'streptavidin-gold-old'
	feature_loader.connect()
	for v in d_new['feature_mask']:
		print (d_new['feature_loader'].feature_names[v])

	compare(models, comparison_name, comparison_mutants_file,results_folder, splits = splits)


