import pandas as pd

from mutedpy.experiments.streptavidin.active_learning.experiment_design import generate_elems
from mutedpy.experiments.streptavidin.active_learning.compare_different_models import load_model
from mutedpy.experiments.streptavidin.streptavidin_loader import load_total
from mutedpy.utils.sequences.sequence_utils import add_variant_column
from mutedpy.protein_learning.safety.safety_model import ResidueBlockSafetyModel, GPSafetyModel
if __name__ == "__main__":
    # params
    N = 120

    # load total data
    x, y, dts = load_total()

    # load safety
    safety = pd.read_csv("OD_model/lists/safety.csv")
    s_columns = ["safety_1", "safety_2"]

    # generate Geometric safe and mean
    #######################################
    model_params = "Geometric/params/final_model_params_Jan_05.p"
    GP, embed = load_model(model_params, "second round fit")

    # load predictions
    predictions = pd.read_csv("Geometric/lists/predictions-geo_Jan_05.csv")
    predictions = add_variant_column(predictions)
    predictions = predictions.merge(safety, how='inner', on='variant')

    safe = generate_elems(predictions,N,already_known_variants=dts, type = 'lcb', name = 'safe', GP= GP, embed=embed, safety_column_names=s_columns)
    dts = pd.concat([dts,safe])
    balanced = generate_elems(predictions, N,already_known_variants=dts, type='mean', name='balanced', GP=GP, embed=embed, safety_column_names=s_columns)

    aa_model_suggestions = pd.concat([safe,balanced])
    aa_model_suggestions.to_csv("Geometric/lists/safe+balanced-geo_Jan_05.csv")
    aa_model_suggestions.to_html("Geometric/lists/safe+balanced-geo_Jan_05.html")

