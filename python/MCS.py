import sys; sys.path.insert(0, './python')
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from main import Main
import shutil
import glob
import pdb
file_marker = "[" + str(os.path.basename(__file__)) + "]: "
random.seed(123)
np.random.seed(123)


def conduct_meta_mcs(par_file_skeleton, nb_iterations):
    """Conducts a Monte Carlo simulation

    Calls `main.Main()` for various parameter files.
    All parameter files that match `par_file_skeleton` will be used.
    
    Parameters
    ----------
    par_file_skeleton : str
        The skeleton path to parameter files that should be used as parameter 
        files for the Monte Carlo Simulation.
    nb_iterations : int
        Number of iterations for the simulation.
    """
    output_dir = par_file_skeleton.replace('python/parameters/', 'output/')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(file_marker + "Removed old directory {}".format(output_dir))
    try:
        os.mkdir(output_dir)
    except OSError:
        print(file_marker + 
              "Creation of directory {} failed".format(output_dir))
    else:
        print(file_marker + 
              "Successfully created directory {} ".format(output_dir))
    par_files = glob.glob(par_file_skeleton + "*.json")
    print(file_marker + "Conduct simulation for parameter files: ")
    print(*par_files, sep="\n")

    # 1. Conduct the simulation runs with distinct parameter specs
    print(file_marker + "STARTING SIMULATIONS")
    for i in range(len(par_files)):
        print(file_marker + "Running parameter file " + par_files[i])
        full_sim = Main(parameter_filename=par_files[i], 
                        iterations=nb_iterations, 
                        output_folder=output_dir) 
    feather_files = glob.glob(output_dir + "/*.feather")
    print(feather_files)

    # Create adequate file names
    agg_results_filename = output_dir + "/" + \
        output_dir.replace('output/', '') + "_agg.feather"
    agg_vis_filename = agg_results_filename.replace("_agg.feather", "_vis.pdf")

    aggregate_results(feather_files, agg_results_filename)


def aggregate_results(output_files, agg_filename):
    """Aggregates the results from a previous Monte Carlo Simulation

    Takes the results as produced by the previous call of `conduct_meta_mcs()`.
    Therefore, it should always be called from within this function or
    afterwards. It aggregates all the results for the feather files passed as
    an input to this function.

    Parameters
    -----------
    output_files : list
        List with feather files containing the model output to be aggregated.

    agg_filename : str
        Name for the file in which to save aggregated results.
    """

    print(file_marker + "STARTING AGGREGATION")
    feather_files = output_files

    results = []
    for i in range(len(feather_files)):
        print(file_marker + str(i))
        x = pd.read_feather(feather_files[i])
        results.append(x)
    
    overall_results = pd.concat(results, ignore_index=True, sort=False)
    opt_diff_results = overall_results

    opt_diff_results.reset_index(inplace=True, drop=True)  
    # drop=True: column 'index' gets removed

    opt_diff_results.to_feather(agg_filename)
    print(file_marker + "Aggregated results saved to: " + agg_filename)


if __name__ == '__main__':
    current_dir = os.getcwd()
    if current_dir[-6:] == "python":
        os.chdir("../")
    current_dir = os.getcwd()
    print(file_marker + "Directory for execution is: " + current_dir)
    if len(sys.argv) < 3:
        print(file_marker,
              "Missing skeleton for parameter files.",
              "Usage: python [parameter_skeleton] [nb iterations]")
        print(file_marker + "For example:")
        print(file_marker + "python python/parameters/opt_diff_ 2")
        sys.exit(2)
    par_file_basis = sys.argv[1]
    assert isinstance(par_file_basis, str), \
        "Skeleton for parameter files not given as string, but {}".format(
            type(par_file_basis))
    assert par_file_basis[-5:] != ".json", \
        "Skeleton for parameter file must be skeleton, not .json file"
    number_of_iterations = int(sys.argv[2])
    assert str(number_of_iterations)==sys.argv[2], \
    "Nb of iterations not properly translated into int: {} != {}".format(
        sys.argv[2], number_of_iterations)
    conduct_meta_mcs(par_file_basis, number_of_iterations)
