import numpy as np
import argparse 


from gemmr.sample_analysis.annotated import analyze_model_parameters
from gemmr.estimators.combination import C3A
from gemmr.generative_model.other import SubPopulationCCA
from gemmr.estimators import SVDCCA
from gemmr.generative_model.integrative import RandomJointCovarianceModel
from gemmr.estimators.annotated import AnnotatedMultiviewEstimator

parser = argparse.ArgumentParser()
# setting nargs="*" automatically typecasts to list
parser.add_argument("-px", type=int, nargs="*", help="# features of data set x", default=[4])
parser.add_argument("-pya", type=int, nargs="*", help="#features of data set ya", default=[4])
parser.add_argument("-pyb", type=int,nargs="*", default=[4])
parser.add_argument("-rxa", type=float,nargs="*", default=[.3])
parser.add_argument("-rxb", type=float,nargs="*",default=[.3])
parser.add_argument("-ax", type=float,nargs="*",default=[-1])
parser.add_argument("-aya", type=float,nargs="*",default=[-1])
parser.add_argument("-ayb", type=float,nargs="*",default=[-1])
parser.add_argument("-exa_mix", type=float,nargs="*",default=[.9])
parser.add_argument("-n_per_ftr2s", type=int, nargs="*", default=[4])
parser.add_argument("-n_per_ftrs", type=int, nargs="*", default=[4, 16])
parser.add_argument("-iteration", type=int)
parser.add_argument("-dirname", type=str)

args = vars(parser.parse_args())

n_per_ftrs = args.pop('n_per_ftrs')
n_per_ftr2s = args.pop('n_per_ftr2s')
iteration = args.pop('iteration')
dirname = args.pop('dirname')
#possibly change this to str: float and use it in fixed_params arg of analyze_model_paramters
fixed_params_dict = {
    "mix_component":[-1,],
    "n_pc_skip": [0],
    "random_state": np.arange(10),
    #"n_components_a": [15],
}

#args that aren't n_per_ftrs or n_per_ftr2s or iteration number
varying_params_dict = args

common_truth = SubPopulationCCA(normalize_weights=True)

ialgs = [
    AnnotatedMultiviewEstimator('cca', SVDCCA(), SubPopulationCCA()),
    AnnotatedMultiviewEstimator('c3a', C3A(), C3A()),
]
# iterate over each parameter combination and run estimator for each possible set of parameters
ds = analyze_model_parameters(
    algs=ialgs,
    GM=RandomJointCovarianceModel,
    params= fixed_params_dict | varying_params_dict,
    common_truth=common_truth, #algorithm to use as ground truth to compare C3A to 
    n_per_ftrs=n_per_ftrs, # number of subjects per feature (X + Ya) in subject set 1
    n_per_ftr2s=n_per_ftr2s, # number of subjects per feature (X+ Yb) in data set 2
    n_rep=5, #number of times to generate dataset and run CCA - ideally 10 -100 times
)

#ds_mean = ds.sel(mode=0).mean('random_state').mean('rep').mean('n_pc_skip')
filename="{}/ds_{}.nc".format(dirname, str(iteration))
ds.to_netcdf(filename)