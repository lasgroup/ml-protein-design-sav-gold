from mutedpy.experiments.streptavidin.active_learning.compare_different_models import compare_from_params
from mutedpy.protein_learning.active_learning.generate_predictions import load_model

splits = 10
comparison_name = "plots/amino-acid-comparison-DEC_2022.png"
comparison_mutants_file = '../AA_model/lists/random_mutants.csv'

model, embed, d = load_model("params/final_model_params.p", model_params_return=True)
model_new, embed_new, d_new = load_model("params/dec_2022_finalfinal_model_params.p", model_params_return=True)
model_old, embed_old, d_old = load_model("params/old_dec_2022_finalfinal_model_params.p", model_params_return=True)
model_aa_old, embed_aa_old, d_aa_old = load_model("../AA_model/params/final_model_params.p", model_params_return=True, vintage = True)
model_aa_new, embed_aa_new, d_aa_new = load_model("../AA_model/params/dec_2022_finalfinal_model_params.p", model_params_return=True, vintage = False)

feature_loader = d['feature_loader']
feature_loader.feature_loaders[0].Embedding.project = 'streptavidin-gold-old'
feature_loader.connect()
for v in d_new['feature_mask']:
    print(d_new['feature_loader'].feature_names[v])

compare_from_params([model,model_new,model_old,model_aa_old,model_aa_new],[embed, embed_new,embed_old,embed_aa_old,embed_aa_new],None, N = 1000)