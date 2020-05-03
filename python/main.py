import sys; sys.path.insert(0, './python')
from model import Model
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pdb
import itertools
from time import localtime, strftime
file_marker = "[" + str(os.path.basename(__file__)) + "]: "


class Main:
    """The main class for the simulations
    
    This class is used to process a single json parameter file: it loads the
    parameters from the json file and then creates a given number of single
    model instances, runs them and aggregates their results. It can be 
    called directly from the command line or via `MCS.py`. The latter option
    is used when more than one json files need to be processed.
    
    Usually it runs the model `n` times with the same parameter constellation.
    If, however, one or more parameters in the json files are given as lists,
    the model instance is run `n` times for each possible combination of the
    parameters given as lists. This is useful for parameter sweeps but quickly
    becomes computationally very intensive.
    
    The file creates the model output in the .feather format. The output is
    saved both for each model instance as well as in its aggregated form.
    Usually, the single model results can also be removed. But this step is
    not done by this script itself.
    """

    def __init__(self, parameter_filename, iterations, output_folder):
        print(file_marker + "Parameter file: " + parameter_filename)
        print(file_marker + "Called __init__ of main.py")
        parameters = json.load(open(parameter_filename))
        assert isinstance(parameters, dict), "Parameter file must be dict, \
        not {}".format(type(parameters))
        assert isinstance(iterations, int), "Nb of iterations must be int, \
        not {}".format(type(iterations))

        self.parameters = parameters
        
        if list in [type(i) for i in self.parameters.values()]:
            parameter_keys_with_list_values = \
                [k for k,v in self.parameters.items() if isinstance(v, list)]
            value_contents = \
                [self.parameters[i] for i in parameter_keys_with_list_values]
            combis = list(itertools.product(*value_contents)) 
            info_file_name = \
                output_folder + "/" + parameter_filename[18:-5] + "combis.txt"
            with open(info_file_name, 'w') as f:
                f.write(
                    f'Keys with varying inputs: {parameter_keys_with_list_values}')
                f.write(
                    f'\nUsed combinations: {combis}')
                f.write(
                    f'\nTotal number of combinations: {len(combis)}')
                f.write(
                    f'\nStarted on: {strftime("%Y-%m-%d %H:%M:%S", localtime())}')

            for cc in range(len(combis)):
                print("Start combi {}/{}".format(cc, len(combis)))
                comb = combis[cc]
                for k in range(len(parameter_keys_with_list_values)):
                    self.parameters[parameter_keys_with_list_values[k]] = comb[k]
                    if parameter_keys_with_list_values[k] == "C_control_1":
                        self.parameters["C_control_2"] = comb[k]
                    if parameter_keys_with_list_values[k] == "D_control_1":
                        self.parameters["D_control_2"] = comb[k]
                        
                self.base_name = parameter_filename[18:-5] + "comb_" + str(cc)
                self.outcome_filename = output_folder + "/" + self.base_name + ".feather"

                self.results = []
                self.results_dist = []
                self.current_id = 1
                while self.current_id <= iterations:
                    print(file_marker + "Start iteration ", self.current_id, " of ", 
                        iterations)
                    current_model = Model(
                        identifier=self.current_id,
                        parameters=self.parameters,
                        base_name=self.base_name, 
                        output_filename=self.outcome_filename
                        )
                    current_model.run()
                    model_results = current_model.return_results()
                    self.results.append(model_results["dynamics"])
                    self.current_id += 1
                self.results_frame = pd.concat(self.results, ignore_index=True)
                self.results_frame = self.aggregate_results(self.results_frame)
                self.save_data()
        else:
            self.base_name = parameter_filename[18:-5]
            self.outcome_filename = output_folder + "/" + self.base_name + ".feather"

            self.results = []
            self.results_dist = []
            self.current_id = 1
            while self.current_id <= iterations:
                print(file_marker + "Start iteration ", self.current_id, " of ", 
                    iterations)
                current_model = Model(
                    identifier=self.current_id,
                    parameters=parameters,
                    base_name=self.base_name, 
                    output_filename=self.outcome_filename
                    )
                current_model.run()
                model_results = current_model.return_results()
                self.results.append(model_results["dynamics"])
                self.current_id += 1
            self.results_frame = pd.concat(self.results, ignore_index=True)
            self.results_frame = self.aggregate_results(self.results_frame)
            self.save_data()

    def aggregate_results(self, full_data_frame):
        """
        Takes the results for individual model runs and aggregates those with
        the same parameter specification.
        
        Attributes
        ----------
        full_data_frame : pd.DataFrame
            A data frame containing the results of all model runs.
            Usually created via: `pd.concat(self.results, ignore_index=True)`
            This should be run automatically one line before calling this 
            function.

        Returns
        -------
        pd.DataFrame
        """
        lower_quant = 0.15
        higher_quant = 0.85
        mid_quant = 0.5
        def quant_low(x):
            return np.quantile(x, lower_quant)
        def quant_high(x):
            return np.quantile(x, higher_quant)
        def quant_mid(x):
            return np.quantile(x, mid_quant)
        
        grouping_parameters = ["t", "n_hawalas", "n_places", "int_pp", "int_ph", 
                               "p_gossip_rej", "p_cheat_rej", "lag_gossip", 
                               "selec_perc", "ranking_methods", "init_share_C",
                               "selection_method", "payoff_a", "payoff_b", 
                               "payoff_c", "payoff_d", "tshock", "cshock", 
                               "tshock_value", "cshock_value",
                               "shock", "trust_control", "c_trust", "d_trust",
                               "c_control_1", "c_control_2", "d_control_1",
                               "d_control_2", "pop_growth", "pop_growth_kind",
                               "pop_growth_type", "mistake_prob"]
        full_data_frame_agg = full_data_frame.groupby(grouping_parameters).agg(
           n_c_mean=("n_c", np.mean),
           n_c_sd=("n_c", np.std),
           n_c_low=("n_c", quant_low),
           n_c_high=("n_c", quant_high),
           n_c_mid=("n_c", quant_mid),
           
           n_d_mean=("n_d", np.mean),
           n_d_sd=("n_d", np.std),
           n_d_low=("n_d", quant_low),
           n_d_high=("n_d", quant_high),
           n_d_mid=("n_d", quant_mid),
           
           sh_d_mean=("sh_d", np.mean),
           sh_d_sd=("sh_d", np.std),
           sh_d_low=("sh_d", quant_low),
           sh_d_high=("sh_d", quant_high),
           sh_d_mid=("sh_d", quant_mid),
           
           sh_c_mean=("sh_c", np.mean),
           sh_c_sd=("sh_c", np.std),
           sh_c_low=("sh_c", quant_low),
           sh_c_high=("sh_c", quant_high),
           sh_c_mid=("sh_c", quant_mid),
           
           sh_c_top_mean=("sh_c_top", np.mean),
           sh_c_top_sd=("sh_c_top", np.std),
           sh_c_top_low=("sh_c_top", quant_low),
           sh_c_top_high=("sh_c_top", quant_high),
           sh_c_top_mid=("sh_c_top", quant_mid),
           
           sh_c_low_mean=("sh_c_low", np.mean),
           sh_c_low_sd=("sh_c_low", np.std),
           sh_c_low_low=("sh_c_low", quant_low),
           sh_c_low_high=("sh_c_low", quant_high),
           sh_c_low_mid=("sh_c_low", quant_mid),
           
           sh_d_top_mean=("sh_d_top", np.mean),
           sh_d_top_sd=("sh_d_top", np.std),
           sh_d_top_low=("sh_d_top", quant_low),
           sh_d_top_high=("sh_d_top", quant_high),
           sh_d_top_mid=("sh_d_top", quant_mid),
           
           sh_d_low_mean=("sh_d_low", np.mean),
           sh_d_low_sd=("sh_d_low", np.std),
           sh_d_low_low=("sh_d_low", quant_low),
           sh_d_low_high=("sh_d_low", quant_high),
           sh_d_low_mid=("sh_d_low", quant_mid),
           
           tr_fail_mean=("tr_fail", np.mean),
           tr_fail_sd=("tr_fail", np.std),
           tr_fail_low=("tr_fail", quant_low),
           tr_fail_high=("tr_fail", quant_high),
           tr_fail_mid=("tr_fail", quant_mid),
           
           tr_fail_sh_mean=("tr_fail_sh", np.mean),
           tr_fail_sh_sd=("tr_fail_sh", np.std),
           tr_fail_sh_low=("tr_fail_sh", quant_low),
           tr_fail_sh_high=("tr_fail_sh", quant_high),
           tr_fail_sh_mid=("tr_fail_sh", quant_mid),
           
           tr_rejct_mean=("tr_rejct", np.mean),
           tr_rejct_sd=("tr_rejct", np.std),
           tr_rejct_low=("tr_rejct", quant_low),
           tr_rejct_high=("tr_rejct", quant_high),
           tr_rejct_mid=("tr_rejct", quant_mid),
           
           tr_rlzd_mean=("tr_rlzd", np.mean),
           tr_rlzd_sd=("tr_rlzd", np.std),
           tr_rlzd_low=("tr_rlzd", quant_low),
           tr_rlzd_high=("tr_rlzd", quant_high),
           tr_rlzd_mid=("tr_rlzd", quant_mid),
           
           tr_rlzd_sh_mean=("tr_rlzd_sh", np.mean),
           tr_rlzd_sh_sd=("tr_rlzd_sh", np.std),
           tr_rlzd_sh_low=("tr_rlzd_sh", quant_low),
           tr_rlzd_sh_high=("tr_rlzd_sh", quant_high),
           tr_rlzd_sh_mid=("tr_rlzd_sh", quant_mid),
           
           tr_coops_mean=("tr_coops", np.mean),
           tr_coops_sd=("tr_coops", np.std),
           tr_coops_low=("tr_coops", quant_low),
           tr_coops_high=("tr_coops", quant_high),
           tr_coops_mid=("tr_coops", quant_mid),
           
           tr_explts_mean=("tr_explts", np.mean),
           tr_explts_sd=("tr_explts", np.std),
           tr_explts_low=("tr_explts", quant_low),
           tr_explts_high=("tr_explts", quant_high),
           tr_explts_mid=("tr_explts", quant_mid),
           
           tr_defcts_mean=("tr_defcts", np.mean),
           tr_defcts_sd=("tr_defcts", np.std),
           tr_defcts_low=("tr_defcts", quant_low),
           tr_defcts_high=("tr_defcts", quant_high),
           tr_defcts_mid=("tr_defcts", quant_mid),
           
           int_strgr_mean=("int_strgr", np.mean),
           int_strgr_sd=("int_strgr", np.std),
           int_strgr_low=("int_strgr", quant_low),
           int_strgr_high=("int_strgr", quant_high),
           int_strgr_mid=("int_strgr", quant_mid),
           
           int_prtnr_mean=("int_prtnr", np.mean),
           int_prtnr_sd=("int_prtnr", np.std),
           int_prtnr_low=("int_prtnr", quant_low),
           int_prtnr_high=("int_prtnr", quant_high),
           int_prtnr_mid=("int_prtnr", quant_mid),
           
           int_coprts_mean=("int_coprts", np.mean),
           int_coprts_sd=("int_coprts", np.std),
           int_coprts_low=("int_coprts", quant_low),
           int_coprts_high=("int_coprts", quant_high),
           int_coprts_mid=("int_coprts", quant_mid),
           
           int_dfctrs_mean=("int_dfctrs", np.mean),
           int_dfctrs_sd=("int_dfctrs", np.std),
           int_dfctrs_low=("int_dfctrs", quant_low),
           int_dfctrs_high=("int_dfctrs", quant_high),
           int_dfctrs_mid=("int_dfctrs", quant_mid),
           
           pp_int_c_mean=("pp_int_c", np.mean),
           pp_int_c_sd=("pp_int_c", np.std),
           pp_int_c_low=("pp_int_c", quant_low),
           pp_int_c_high=("pp_int_c", quant_high),
           pp_int_c_mid=("pp_int_c", quant_mid),
           
           pp_int_d_mean=("pp_int_d", np.mean),
           pp_int_d_sd=("pp_int_d", np.std),
           pp_int_d_low=("pp_int_d", quant_low),
           pp_int_d_high=("pp_int_d", quant_high),
           pp_int_d_mid=("pp_int_d", quant_mid),
           
           eff_temp_mean=("eff_temp", np.mean),
           eff_temp_sd=("eff_temp", np.std),
           eff_temp_low=("eff_temp", quant_low),
           eff_temp_high=("eff_temp", quant_high),
           eff_temp_mid=("eff_temp", quant_mid),
           
           eff_ovrl_mean=("eff_ovrl", np.mean),
           eff_ovrl_sd=("eff_ovrl", np.std),
           eff_ovrl_low=("eff_ovrl", quant_low),
           eff_ovrl_high=("eff_ovrl", quant_high),
           eff_ovrl_mid=("eff_ovrl", quant_mid)
        )
        full_data_frame_agg.reset_index(inplace=True)
        return full_data_frame_agg        

    def save_data(self):
        """Saves the data to a pd.dataFrame"""
        print(file_marker + "Start saving data...", end="")
        if self.outcome_filename[-8:] != ".feather":
            self.outcome_filename += ".feather"
        dist_outcome_filename = \
            self.outcome_filename.replace(".feather", "_dist.feather")
        self.results_frame.reset_index(drop=True).to_feather(self.outcome_filename)
        print(file_marker + "complete!")
        print(file_marker + "Outcome saved in: {}".format(self.outcome_filename))

    @staticmethod
    def get_colors(cm_name, nb_cols):
        """Gives a list of color codes from a given color map.

        Parameters
        ----------
        cm_name : str
            Name of a color map used in matplotlib.pyplot
        nb_cols : int
            Number of different colors to be returned.

        Returns
        -------
        list
            A list with `nb_cols` color codes from color map `cm_name`.
        """
        cmap_object = plt.get_cmap(cm_name)
        col_codes = cmap_object(np.linspace(1, 256, nb_cols) / 100)
        return col_codes
    