import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import argparse
import os
import numpy as np
import pandas as pd

# loaders
from mutedpy.utils.loaders.loader_basel import BaselLoader
from mutedpy.protein_learning.embeddings.esm_embedding import ESMEmbedding

# utils
from mutedpy.utils.protein_operator import ProteinOperator
from mutedpy.utils.sequences.sequence_utils import drop_neural_mutations, order_mutations
from mutedpy.experiments.streptavidin.streptavidin_loader import load_first_round, load_total, load_od_data

# models
from mutedpy.protein_learning.embeddings.amino_acid_embedding import AminoAcidEmbedding
from mutedpy.protein_learning.regression.regression_ards_geometry import LassoRFFeatureSelectorARDGeometric, \
    LassoFeatureSelectorARDGeometric
from mutedpy.protein_learning.regression.regression_ards import ARDModelLearner, ProteinKernelLearner
from mutedpy.protein_learning.featurizers.feature_loader import ProteinFeatureLoader, AddedProteinFeatureLoader
from mutedpy.protein_learning.featurizers.feature_selector import LassoFeatureSelector,RFFeatureSelector, ElasticNetFeatureSelector,DummyFeatureSelector

# evaluator
from mutedpy.protein_learning.evaluation.evaluator import Evaluator

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='run streptavidin analysis')
    parser.add_argument('--kernel',            action='store', help='', default = 'ard_matern')
    parser.add_argument('--topk',              action='store', help='', default = 30, type = int)
    parser.add_argument('--feature_selector',  action='store', help='', default = 'rf', type = str)
    parser.add_argument('--aminoacid',         action='store', help='', default=1, type=int)
    parser.add_argument('--aminoacid_full',    action='store', help='', default=1, type=int)
    parser.add_argument('--aminoacid_pair',    action='store', help='', default=1, type=int)
    parser.add_argument('--additive',    action='store', help='', default=0, type=int)
    parser.add_argument('--rosetta',           action='store', help='', default = 1, type = int)
    parser.add_argument('--esm',           action='store', help='', default = 0, type = int)
    parser.add_argument('--results_folder',            action='store', help='', default = "results_strep", type = str)
    parser.add_argument('--geometric',         action='store', help='', default = 1, type = int)
    parser.add_argument('--model',             action='store', help='', default = 'ARD', type = str)
    parser.add_argument('--project_geo',       action='store', help='', default = 'streptavidin-gold-def', type = str)
    parser.add_argument('--project_ro',        action='store', help='', default='streptavidin-gold-detailed', type=str)
    parser.add_argument('--restarts',          action='store', help='', default = 3, type = int)
    parser.add_argument('--maxiters',          action='store', help='', default = 200, type = int)
    parser.add_argument('--splits',            action='store', help='', default = 5, type = int)
    parser.add_argument('--special_identifier',action='store', help='', default = '', type = str)
    parser.add_argument('--cores_split',       action='store', help='', default = 5, type = int)
    parser.add_argument('--njobs',             action='store', help='', default = 10, type = int)
    parser.add_argument('--transformation',    action='store', help='', default='log', type=str)
    parser.add_argument('--split_loc',         action='store', help='', default="../splits/random_splits.pt", type=str)
    parser.add_argument('--target',            action='store', help='', default = 'evaluate', type = str)
    parser.add_argument('--final_dir',         action='store', help='', default = './', type = str)
    parser.add_argument('--data',              action='store', help='', default = 'first', type = str)
    parser.add_argument('--model_selection_data',action='store', help='', default = 'all', type = str)
    parser.add_argument('--prespecified_sigma', action='store', help='', default = None)
    parser.add_argument('--subsample',           action='store', help='', default = 1., type = float)
    parser.add_argument('--seed',           action='store', help='', default = 1., type = int)
    parser.add_argument('--scaler', action='store', help='', default='default', type=str)
    args = parser.parse_args()

    ########################
    #### Data loading    ###
    ########################
    if args.data == "first":
        print ("Loading first round data only.")
        x, y, total_dts = load_first_round(transformation = args.transformation)

    elif args.data == "total":
        print ("Learning the total dataset. ")
        x, y, total_dts = load_total()


    elif args.data == "subsample":

        x, y, total_dts = load_first_round(transformation=args.transformation)

        # subsample data
        np.random.seed(args.seed)

        ind = np.random.choice(np.arange(0,x.size()[0],1),size = int(args.subsample*x.size()[0]))

        x = x[ind,:]
        y = y[ind,:]

    elif args.data == "OD":
        print("Learning OD dataset.")
        x, y, total_dts = load_od_data()

    ###########################
    ### initialize the model ##
    ###########################

    models = []
    restarts = args.restarts
    maxiter = args.maxiters
    splits = args.splits

    special_identifier = args.special_identifier + "topk" + str(args.topk)
    params = {'kernel': args.kernel, 'topk': args.topk, 'additive': args.additive}

    ########################
    #### Feature loading ###
    ########################
    total_features_list = []
    total_features_caller_list = []
    total_features_param_list = []
    # geometric features
    if os.path.dirname(__file__) == "":
        path_dir = "./"
    else:
        path_dir = os.path.dirname(__file__) +"/"
    print ("PATH DIR:", path_dir)

    if args.geometric:
        feature_loader_caller = ProteinFeatureLoader
        feature_loader_param = {'data':None,
                                'pandas_data':total_dts,
                              'aa_positions':[111, 112, 118, 119, 121],
                              'wild_type':"TSNAK",
                              'datatype':'mongo+preload',
                              'data_folder':path_dir+'/../data/',
                              'server':'matroid2.inf.ethz.ch:27017',
                              'credentials':path_dir+'/../../../server/credentials.pass',
                              'database':'experiment-features-selected',
                              'project':args.project_geo}

        feature_loader = ProteinFeatureLoader(**feature_loader_param)

        total_features_list.append(feature_loader)
        total_features_caller_list.append(feature_loader_caller)
        total_features_param_list.append(feature_loader_param)

    if args.aminoacid:
        feature_loader2 = AminoAcidEmbedding(data=path_dir + "/../data/amino-acid-features.csv", projection = path_dir+ "/../data/embedding-dim5-demean-norm.pt", proto_names = 5)
        feature_loader_caller2 = AminoAcidEmbedding
        feature_loader_param2 = {'data':path_dir + "/../data/amino-acid-features.csv",
                                'projection': path_dir+"/../data/embedding-dim5-demean-norm.pt",
                                 'proto_names': 5}
        #feature_loader2.load_projection(path_dir + "/../data/embedding-dim5-demean-norm.pt")
        #feature_loader2.set_proto_names(5)

        total_features_caller_list.append(feature_loader_caller2)
        total_features_param_list.append(feature_loader_param2)
        total_features_list.append(feature_loader2)

    # rosetta features
    if args.rosetta:
        feature_loader_caller3 = ProteinFeatureLoader
        feature_loader_param3 = {'data':None,'pandas_data':total_dts,
                                          'aa_positions':[111, 112, 118, 119, 121],
                                          'wild_type':"TSNAK",
                                          'datatype':'mongo+preload',
                                          'data_folder':path_dir+'/../data/',
                                          'server':'matroid2.inf.ethz.ch:27017',
                                          'credentials':path_dir+'/../../../server/credentials.pass',
                                          'database':'experiment-features-selected',
                                          'project':args.project_ro}
        feature_loader3 = ProteinFeatureLoader(**feature_loader_param3)

        total_features_caller_list.append(feature_loader_caller3)
        total_features_param_list.append(feature_loader_param3)
        total_features_list.append(feature_loader3)


    if args.esm:
        feature_loader_caller3 = ProteinFeatureLoader
        feature_loader_param3 = {'data':None,'pandas_data':total_dts,
                                          'aa_positions':[111, 112, 118, 119, 121],
                                          'wild_type':"TSNAK",
                                          'datatype':'mongo+preload',
                                          'data_folder':path_dir+'/../data/',
                                          'server':'matroid2.inf.ethz.ch:27017',
                                          'credentials':path_dir+'/../../../server/credentials.pass',
                                          'database':'experiment-features',
                                          'project':"streptavidin-esm2",
                                        'embedding_name':'embedding-mean'}
        feature_loader3 = ProteinFeatureLoader(**feature_loader_param3)

        total_features_caller_list.append(feature_loader_caller3)
        total_features_param_list.append(feature_loader_param3)
        total_features_list.append(feature_loader3)


    ###########################
    #### Feature selector   ###
    ###########################
    if args.feature_selector == "lasso":
        selector_caller = LassoFeatureSelector
        selector_params = {'njobs':args.njobs}
        selector = LassoFeatureSelector(njobs=args.njobs)
    elif args.feature_selector == "rf":
        selector_caller = RFFeatureSelector
        selector_params = {'njobs':args.njobs}
        selector = RFFeatureSelector(njobs=args.njobs)
    elif args.feature_selector == "none":
        selector_caller = DummyFeatureSelector
        selector_params = {'njobs':args.njobs}
        selector = DummyFeatureSelector()
    elif args.feature_selector == "elastic":
        selector_caller = ElasticNetFeatureSelector
        selector_params = {'njobs': args.njobs}
        selector = ElasticNetFeatureSelector(njobs=args.njobs)
    ###########################
    #### Model select     #####
    ###########################

    if args.model == "ARD":
        model_caller = ARDModelLearner
    elif args.model == "linear":
        model_caller = ProteinKernelLearner

    model_name = args.special_identifier+"_".join([args.feature_selector, str(args.rosetta),
                                                              str(args.geometric), str(args.aminoacid),str(args.esm),
                                                              str(args.topk),args.project_geo,args.kernel,args.model,
                                                                args.model_selection_data, args.transformation, str(args.additive),"noise" + str(args.prespecified_sigma)])

    default_params = {'restarts': restarts,
                      'results_folder': "../"+args.results_folder+"/"+model_name+"/",
                      "maxiter": maxiter,
                      'feature_loader_caller_list': total_features_caller_list,
                      'feature_loader_params_list':total_features_param_list,
                      'feature_selector_caller': selector_caller,
                      'feature_selector_params': selector_params,
                      'cores': args.cores_split,
                      'njobs': args.njobs,
                      'prespecified_sigma': float(args.prespecified_sigma) if args.prespecified_sigma is not None else None,
                      'model_selection_all': True if args.model_selection_data == "all" else False
                      }


    evaluator = Evaluator(model_name,x,y,model_caller,{**default_params, **params})

    # evaluate on metric and save
    if args.target == "evaluate":
        # create folder for results_strep
        try:
            if not os.path.exists(path_dir+default_params['results_folder']):
                os.mkdir(path_dir+default_params['results_folder'])
            else:
                print("Folder exists: " +path_dir+default_params['results_folder'])
        except:
            print("Folder creation failed: "+path_dir+default_params['results_folder'])


        evaluator.evaluate_metrics_on_splits(no_splits=splits,
                                     split_location=args.split_loc,
                                     special_identifier='')

    elif args.target == "subsampling_analysis":
        model = model_caller(**{**default_params, **params})
        model.add_data(x, y)
        model.production()
        model.fit(save_loc=args.final_dir + "params.np")
        model.save_model(id="sub", save_loc=args.final_dir + "subsampling"+str(args.subsample)+"_"+str(args.seed)+".p")

    elif args.target =="final_model":
        model = model_caller(**{**default_params, **params})
        model.add_data(x, y)
        model.production()
        model.fit(save_loc=args.final_dir+"params.np")
        model.save_model(id="final", save_loc=args.final_dir+"final_model_params.p")