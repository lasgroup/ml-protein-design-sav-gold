from mutedpy.experiments.streptavidin.active_learning.calculate_predictions import generate_all_predictions, add_embelishments

if __name__ == "__main__":
	positions = [111,112,118,119,121]
	parent = 'TSNAK'

	# model_params = "AA_model/params/final_model_params.p"
	# new_mutants = generate_all_predictions(model_params,positions,parent)
	# new_mutants = add_embelishments(new_mutants)
	# new_mutants.to_csv('AA_model/lists/predictions-aa.csv')

	model_params = "Geometric/params/final_model_params_Jan_05.p"
	new_mutants = generate_all_predictions(model_params,positions,parent)
	new_mutants = add_embelishments(new_mutants)
	new_mutants.to_csv('Geometric/lists/predictions-geo_Jan_05.csv')

	# model_params = "Rosetta_init/params/final_model_params.p"
	# new_mutants = generate_all_predictions(model_params,positions,parent)
	# new_mutants = add_embelishments(new_mutants)
	# new_mutants.to_csv('Rosetta_init/lists/predictions-ro.csv')
