import pandas as pd
import torch
from mutedpy.experiments.streptavidin.active_learning.calculate_predictions import add_embelishments
from mutedpy.experiments.streptavidin.active_learning.experiment_design import generate_elems
from mutedpy.experiments.streptavidin.active_learning.compare_different_models import load_model
import pandas as pd
from mutedpy.experiments.streptavidin.streptavidin_loader import load_full, load_total
import numpy as np
from dppy.finite_dpps import FiniteDPP
from mutedpy.protein_learning.embeddings.amino_acid_embedding import AminoAcidEmbedding
from mutedpy.protein_learning.featurizers.feature_loader import ProteinFeatureLoader,AddedProteinFeatureLoader
from mutedpy.utils.sequences.sequence_utils import get_number_of_muts

dts = []
workers = 320
for worker in range(workers):
    file_name = "output/output" + str(worker) + ".csv"
    try:
        d = pd.read_csv(file_name)

        dts.append(d)
    except:
        print (file_name, "failed.")

dts = pd.concat(dts)
dts['no_sites'] = get_number_of_muts(dts)
dts = dts[dts['no_sites'] <= 8]
predictions = add_embelishments(dts)
print (predictions.head(2))
predictions.to_csv("Geometric/lists/predictions-geo_Jan_05.csv")