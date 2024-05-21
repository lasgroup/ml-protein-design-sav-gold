from mutedpy.experiments.streptavidin.active_learning.compare_different_models import load_model_and_mean_std, load_model
import pickle
import torch
from mutedpy.utils.sequences.sequence_utils import generate_all_combination, generate_random_mutations,from_mutation_to_variant,from_variant_to_integer
import argparse
import os
from openTSNE import TSNE

parser = argparse.ArgumentParser(description='Generate a database.')
parser.add_argument('--worker_name', action='store', help='name', default='my worker')
parser.add_argument('--embedding', action='store', help='location', default='embAA.pt')
parser.add_argument('--xtest', action='store', help='location', default='embx.pt')
parser.add_argument('--input', action='store', help='', default='input.txt')
parser.add_argument('--model_loc', action='store', help='',
                    default="../active_learning_2/AA_model/params/final_model_params.p")
parser.add_argument('--model_name', action='store', help='', default='AA')
parser.add_argument('--stage', action='store', help='', type=int, default=1)
parser.add_argument('--njobs', action='store', help='', type=int, default=30)
parser.add_argument('--ouput_folder', action='store', help='', type=str, default="/cluster/project/krause/mmutny/clustering/")

parser.add_argument('--no_clusters', action='store', help='', type=int, default=1)
parser.add_argument('--load_only', action='store', help='', type=int, default=0)

args = parser.parse_args()


if __name__ == "__main__":
    print(os.getcwd())
    params = args.model_loc
    model_name = args.model_name
    worker_name = args.worker_name
    no_clusters = args.no_clusters

    # load dataset
    #x, y, d = load_full()

    # load the model
    model, embed, model_params = load_model(params, "name", model_params_return =True)

    # model parameters
    model_params = pickle.load(open(params, "rb"))
    feature_loader = model_params["feature_loader"]
    feature_mask = model_params['feature_mask']

    # load the embedding
    embedding_name = args.embedding
    xtest_name = args.xtest
    ard_gamma = model_params['kernel_object'].params['ard_gamma']

    if os.path.exists(embedding_name) and args.load_only == 0:

        print("loading embedding")
        #xtest = torch.load(xtest_name)
        phitest = torch.load(embedding_name)

        print (phitest.size())
    elif args.load_only == 0:

        # get a random subsample of the space
        print("generating mutation space.")
        variants = generate_all_combination([111, 112, 118, 119, 121], 'TSNAK')
        print("all variant generated")
        # variants = generate_random_mutations(32000, [111, 112, 118, 119, 121], 'TSNAK')
        xtest = from_variant_to_integer(from_mutation_to_variant(variants))

        print("integrers")
        phitest = embed(xtest)
        phitest = torch.einsum('ij,j->ij',  phitest,1./ard_gamma)

        torch.save(xtest,xtest_name)
        torch.save(phitest,embedding_name)

    for per in [120, 150, 200]:
        tsne = TSNE(
            perplexity=per,
            metric="euclidean",
            n_jobs=args.njobs,
            random_state=42,
            verbose=True
            #neighbors= 'pynndescent'
        )
        X_embedded = tsne.fit(phitest.detach().numpy())
        pickle.dump(X_embedded, open(args.ouput_folder+"embAA-cluster_"+str(per)+".pickle","wb"))

    # embedding