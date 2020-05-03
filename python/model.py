import sys; sys.path.insert(0, './python')
import os
import numpy as np
import pandas as pd
import pdb
from operator import methodcaller
from agent import Agent
from strategies import Strategy
from population_generator import PopulationGenerator
import matplotlib.pyplot as plt
file_marker = "[" + str(os.path.basename(__file__)) + "]: "

# TODO Integrate growth in all parameter files and remove try-except block

class Model:
    """The instance of a single model

    This class represents a single model instance as described in the main 
    paper. It is usually instantiated via `Main` since many model runs are
    analyzed via Monte Carlo simulations. These are implemented in `Main`.
    """
    def __init__(self, identifier, parameters, base_name, output_filename):
        """Creates a single model instance
        
        [extended_summary]
        
        Parameters
        ----------
        identifier : int
            An identifier for the model instance. Will be created within Main
        parameters : dict
            A dictionary with parameters. Will be created within Main from the
            relevant json file
        base_name : str
            The name of the model used for saving results. Will be created in
            `Main` automatically from the name of the json file containing the
            parameters
        output_filename : str
            The name of the output files. Will be created via `Main`.
        """
        print(file_marker + "Initializing model " + str(identifier) + "...")
        self.id = identifier
        self.base_name = base_name
        
        # Set general parameters
        assert isinstance(parameters, dict), \
            "Parameters not given as dict but {}".format(type(parameters))
        self.parameters = parameters
        
        self.one_region = \
            True if self.parameters["number_of_places"] == 1 else False
        self.outputfile_name = output_filename

        self.timestep = 0
        self.stop = False
        self.strategy_class = Strategy(self)
        assert isinstance(self.strategy_class, Strategy), \
            'The strategy object is not properly assigned!'
        #TODO! Integrate into parameter files, then remove try-except        
        try:
            self.initial_share_ALL_C = self.parameters["init_share_ALL_C"]
        except KeyError:
            self.initial_share_ALL_C = \
                self.parameters["distribution_of_strategies"]['ALL_C']
        assert 0.0<=self.initial_share_ALL_C<=1.0, \
            "Wrong initial share for coops given: {}".format(
                self.initial_share_ALL_C)
        self.initial_share_ALL_D = 1.0 - self.initial_share_ALL_C
        
        self.popgen = PopulationGenerator(self.parameters, self)
        self.agents = self.popgen.get_agents()
        self.places = self.popgen.get_places()

        if str(self.parameters["tc_shock"]) == "no":
            self.cshock = str(self.parameters["c_shock"])
            self.tshock = str(self.parameters["t_shock"])
        else:
            self.cshock = str(self.parameters["tc_shock"])
            self.tshock = str(self.parameters["tc_shock"])
        self.cshock_value = str(self.parameters["c_shock_value"])
        self.tshock_value = str(self.parameters["t_shock_value"])
        self.cshock_has_taken_place = False
        self.tshock_has_taken_place = False
        
        self.coop_agents_trust = self.parameters["C_trust"]
        self.coop_agents_control_1 = self.parameters["C_control_1"]
        self.coop_agents_control_2 = self.parameters["C_control_2"]
        self.defec_agents_control_1 = self.parameters["D_control_1"]
        self.defec_agents_control_2 = self.parameters["D_control_2"]
        
        #TODO! Integrate into parameter files, then remove try-except
        try:
            self.pop_growth = self.parameters["pop_growth"]
            self.pop_growth_kind = self.parameters["pop_growth_kind"]
            self.pop_growth_type = self.parameters["pop_growth_type"]
        except KeyError:
            self.pop_growth = 0.0
            self.pop_growth_kind = "NA"
            self.pop_growth_type = "NA"
        
        try:
            self.mistake_prob = self.parameters["mistake_prob"]
        except KeyError:
            self.mistake_prob = 0.0

        # ----------------------------------BEGIN STATUS VARIABLES--------------------------------------------------
        self.connections_dict = dict()
        self.reputations_dict = dict()

        self.wealth = []
        self.nb_of_ALL_C = []
        self.nb_of_ALL_D = []
        self.share_of_ALL_C = []
        self.share_of_ALL_D = []
        self.share_ALL_C_top = []
        self.share_ALL_C_low = []
        self.share_ALL_D_top = []
        self.share_ALL_D_low = []
        self.interactions_by_defectors = []
        self.interactions_by_cooperators = []
        self.pp_interactions_by_cooperators = []
        self.pp_interactions_by_defectors = []

        self.failed_transactions_current = 0
        self.realized_transactions_current = 0
        self.cooperative_transactions_current = 0
        self.exploitations_current = 0
        self.defective_transactions_current = 0
        self.interactions_with_partners_current = 0
        self.interactions_with_strangers_current = 0
        self.interactions_by_defectors_current = 0
        self.interactions_by_cooperators_current = 0
        self.current_t_wealth = 0.0

        self.failed_transactions = []
        self.rejected_transactions = []
        self.realized_transactions = []
        self.realized_transactions_share = []
        self.failed_transactions_share = []
        self.cooperative_transactions = []
        self.exploitations = []
        self.defective_transactions = []
        self.interactions_with_partners = []
        self.interactions_with_strangers = []
        self.temporal_efficiency = []
        self.overall_efficiency = []

        self.init_interaction = [{'chosen_ag_nb': np.nan, 
                                  'strat_chosen': np.nan, 
                                  'partner_ag_nb': np.nan,
                                  'strat_partner': np.nan}]

        self.connections_agent_one = []
        self.connections_agent_two = []
        self.connections_list = []
        self.connection_list_old = []
        self.interaction_matrix = np.zeros([len(self.agents), len(self.agents)])
        self.rejection_matrix = np.zeros([len(self.agents), len(self.agents)])
        self.fails = len(self.agents) * [0]

        # ----------------------------------END STATUS VARIABLES-----------------------------------------------------
        
    def run(self):
        """Runs the model for a given number of time steps
        
        Runs the model as described in the main paper.
        The first time step starts immediately with recording the strategy 
        distribution and the state variables. Otherwise, the procedure is 
        as follows: 
        
        First, it is checked whether trust or control shocks take place, in 
        which case the respective methods of the `Agent` class is called.
        Second, the function `self.update` conducts the interactions phase.
        Third, the function `self.selection_strats` implements the learning
        and selection phase. 
        Fourth, the functions `self.record_strats` records the strategies 
        currently used by the agents and `self.report` saves the current 
        values of all state variables.
        Fifth, if necessary, population growth takes place via the function
        `self.population_growth`.
        """
        params = self.parameters
        for i in range(0, params['number_of_timesteps']):
            self.timestep = i
            print('Round {} of {} for file {} (Run {})'.format(
                str(self.timestep), str(params['number_of_timesteps']-1), 
                str(self.outputfile_name), str(self.id)), end="\r")
            if self.timestep == 0:
                self.record_strats()
                self.report(self.timestep, initial_round=True)
            self.strategy_class.reset_counters()

            """Shocks-------------------------------------------------------"""
            if str(i) == self.tshock:
                print("Trust shock happens in t={}".format(i))
                for a in self.agents:
                    a.shock("trust", self.parameters["t_shock_value"])
                self.tshock_has_taken_place = True
                self.coop_agents_trust = self.parameters["t_shock_value"]

                
            if str(i) == self.cshock:
                print("Control shock happens in t={}".format(i))
                for a in self.agents:
                    a.shock("control", self.parameters["c_shock_value"])
                self.cshock_has_taken_place = True
                self.coop_agents_control_1 = self.parameters["c_shock_value"]
                self.coop_agents_control_2 = self.parameters["c_shock_value"]
            """End shocks---------------------------------------------------"""

            self.update()
            self.selection_strats()
            self.record_strats()
            self.strategy_class.record_counters(i)
            self.report(self.timestep, monitor=False)
            if self.pop_growth>0.0:
                self.population_growth()
            
    def update(self):
        """Implements the interaction phase of the model.
        
        This implements the interaction phase as described in the main paper.
        The procedure is as follwos:

            (1) Create random demand from one region into another
            (2) Select a Hawaladar in the first region
            (3) Check whether the Hawaladar has a contact in this region. 
                If yes, choose a partner, if not choose a stranger
            (4) The two hawaladars play the PD
        """
        params = self.parameters
        iterations = params["interactions_per_period"]
        current_iteration = 1
        while current_iteration <= iterations:
            # (1) Create random demand
            pair = self.random_demand(one_region=self.one_region)
            # (2) Select a Hawaladar in this region
            chosen_hawaladar = self.select_hawaladar(pair[0])
                        
            # (3) Check whether the Hawaladar has a contact in this region. 
            if chosen_hawaladar.check_connection(pair[1]):
                chosen_partner = chosen_hawaladar.get_partner(pair[1])  
                if chosen_partner is not False:
                    self.interactions_with_partners_current += 1

            else:  # Here we must check that it is not the same hawaladar
                chosen_partner = chosen_hawaladar.get_stranger(
                    self.places[pair[1]], params, pair[1])
                if chosen_partner is not False:
                    self.interactions_with_strangers_current += 1

            if chosen_partner is False:
                self.failed_transactions_current += 1  
                self.fails[chosen_hawaladar.ident] += 1
                chosen_hawaladar.wealth.append(float(0.0)) 
                # This is necessry, otherwise calculation of total wealth 
                # received from previous rounds would not work properly

            # (4) The two hawaladars play the PD
            else:
                self.play(chosen_hawaladar, chosen_partner)
            current_iteration += 1

    def return_results(self):
        """Return model outcomes
        
        Should be called by Main.
        
        Returns
        -------
        dict
            Dictionary with pandas.DataFrames. 
            Keys: 
                'dynamics' for the time series results
                'distributions' for the distributions (not yet implemented)
        """
        print(file_marker + "called return_results()")
        params = self.parameters
        ts_len = params['number_of_timesteps'] + 1
        result_dict_dynamics = {
            "id": ts_len * [self.id], 
            "t": range(0, params['number_of_timesteps']+1),
            "n_hawalas" : ts_len * [params["number_of_hawaladars"]], 
            "init_share_C": ts_len * [self.initial_share_ALL_C],
            "n_places" : ts_len * [params["number_of_places"]], 
            "int_pp" : ts_len * [params["interactions_per_period"]], 
            "int_ph" : ts_len * [params["interactions_per_hawaladar"]], 
            "p_gossip_rej" : ts_len * [params["gossip_rejection_prob"]], 
            "p_cheat_rej" : ts_len * [params["cheating_rejection_prob"]], 
            "lag_gossip" : ts_len * [params["time_lag_gossip"]], 
            "selec_perc" : ts_len * [params["selection_percentage"]], 
            "ranking_methods" : ts_len * [params["selector"]], 
            "selection_method" : ts_len * [params["adaptive_strategy_selection"]], 
            "payoff_a" : ts_len * [params["payoffs"]["a"]], 
            "payoff_b" : ts_len * [params["payoffs"]["b"]], 
            "payoff_c" : ts_len * [params["payoffs"]["c"]], 
            "payoff_d" : ts_len * [params["payoffs"]["d"]],
            "cshock" : ts_len * [self.cshock],
            "cshock_value" : ts_len * [self.cshock_value],
            "tshock" : ts_len * [self.tshock],
            "tshock_value" : ts_len * [self.tshock_value],
            "shock" : self.get_shock_description(ts_len),
            "trust_control" : self.get_trust_control_description(ts_len),
            "c_trust": ts_len * [params["C_trust"]],
            "d_trust": ts_len * [params["D_trust"]],
            "c_control_1": ts_len * [params["C_control_1"]],
            "c_control_2": ts_len * [params["C_control_2"]],
            "d_control_1": ts_len * [params["D_control_1"]],
            "d_control_2": ts_len * [params["D_control_2"]],
            "pop_growth": ts_len * [self.pop_growth],
            "pop_growth_kind": ts_len * [self.pop_growth_kind],
            "pop_growth_type": ts_len * [self.pop_growth_type],
            "mistake_prob": ts_len * [self.mistake_prob],
            'n_c': self.nb_of_ALL_C, 
            'n_d': self.nb_of_ALL_D,
            'sh_c': self.share_of_ALL_C, 
            'sh_d': self.share_of_ALL_D,
            'sh_c_top': [self.share_of_ALL_C[0]] + self.share_ALL_C_top,
            'sh_c_low': [self.share_of_ALL_C[0]] + self.share_ALL_C_low,
            'sh_d_top': [self.share_of_ALL_D[0]] + self.share_ALL_D_top,
            'sh_d_low': [self.share_of_ALL_D[0]] + self.share_ALL_D_low,
            'tr_fail': self.failed_transactions,
            'tr_fail_sh': self.failed_transactions_share,
            'tr_rejct': self.rejected_transactions,
            'tr_rlzd': self.realized_transactions,
            'tr_rlzd_sh': self.realized_transactions_share,
            'tr_coops': self.cooperative_transactions,
            'tr_explts': self.exploitations,
            'tr_defcts': self.defective_transactions,
            'int_strgr': self.interactions_with_strangers,
            'int_prtnr': self.interactions_with_partners,
            'int_coprts': self.interactions_by_cooperators,
            'int_dfctrs': self.interactions_by_defectors,
            'pp_int_c': self.pp_interactions_by_cooperators,
            'pp_int_d': self.pp_interactions_by_defectors,
            'eff_temp': self.temporal_efficiency,
            'eff_ovrl': self.overall_efficiency
        }
        result_frame_dynamics = pd.DataFrame.from_dict(result_dict_dynamics)
        
        return_dict = {"dynamics": result_frame_dynamics} 
        return return_dict
    
    def population_growth(self):
        """Implements population growth
        
        There are the following types of population growth:
            1. neutral_growth: the new agents are cooperators or defectors 
                with probability of 50%.
            2. standard_growth: the strategies of the new agents follow the
                initial distribution of the strategies.
            3. directed_growth_1: the strategies of the new agents follow
                the current distribution of the strategies
            4. directed_growth_2: the strategies of the new strategies follow
                the distribution of the best performing agents
            
        Raises
        ------
        InputError
            If the type of population growth is wrongly specified in parfile
        InternalError
            If the new strategies are not ALL_C or ALL_D
        """
        params = self.parameters
        assert isinstance(params["pop_growth"], float)
        assert params["pop_growth"]>0.0
        
        growth = params["pop_growth"]
        if params["pop_growth_kind"] == "constant_number":
            nb_new_agents = int(growth * params["number_of_hawaladars"]) 
        elif params["pop_growth_kind"] == "compounding":
            nb_new_agents = int(growth * len(self.agents))
        else:
            raise InputError("pop_growth_kind has wrong value: {}".format(
                        params["pop_growth_kind"]))
        
        for new_ag in range(nb_new_agents):
            region = int(np.random.choice(list(self.places.keys())))
            assert isinstance(region, int), \
                "Chosen place not int, but {}".format(type(region))
                
            if params["pop_growth_type"] == "neutral_growth":
                strat = np.random.choice(['ALL_D', 'ALL_C'],
                                            p=[0.5, 0.5])
            elif params["pop_growth_type"] == "standard_growth":
                strat = np.random.choice(
                    ['ALL_D', 'ALL_C'],
                    p=[self.initial_share_ALL_D, self.initial_share_ALL_C])
            elif params["pop_growth_type"] == "directed_growth_1":
                strat = np.random.choice(
                    ['ALL_D', 'ALL_C'],
                    p=[self.share_of_ALL_D[-1], self.share_of_ALL_C[-1]])
            elif params["pop_growth_type"] == "directed_growth_2":
                strat = np.random.choice(
                    ['ALL_D', 'ALL_C'],
                    p=[self.share_ALL_D_top[-1], self.share_ALL_C_top[-1]])
            else:
                raise InputError("pop_growth_type has wrong value: {}".format(
                    params["pop_growth_type"]))
                
            if strat == 'ALL_D':
                trust = np.nan
                control_1 = params["D_control_1"]
                control_2 = params["D_control_1"]
                interval = params["D_interval"]
            elif strat == 'ALL_C':
                if self.cshock_has_taken_place:
                    control_1 = params["c_shock_value"]
                    control_2 = params["c_shock_value"]
                else:
                    control_1 = params["C_control_1"]
                    control_2 = params["C_control_2"]
                if self.tshock_has_taken_place:
                    trust = params["t_shock_value"]
                else:
                    trust = params["C_trust"]
                interval = np.NaN
            else:
                raise InternalError(
                    "Chosen strat not ALL_D or ALL_C but {}".format(strat))
            id_new = len(self.agents)
            self.agents.append(Agent(
                id_new, self, region, trust, control_1, control_2, 
                interval, strat, params["partner_selection"]))     
            
    def random_demand(self, one_region=False):
        """
        Returns a tuple with two different regions.
        
        Returns
        -------
        tuple of len 2
            A tuple with the int codes of the regions
        """
        params = self.parameters
        if one_region == True:
            assert params['number_of_places'] == 1, \
                "random_demand told there is 1 region, but {}".format(
                    params['number_of_places'])
            return_tuple = tuple([0, 0])
        else:
            assert params['number_of_places'] > 1, \
                "random_demand told there are many regions, but {}".format(
                    params['number_of_places'])
            list_of_places = list(range(params['number_of_places']))
            first = np.random.choice(list_of_places)
            list_of_places.remove(first)
            second = np.random.choice(list_of_places)
            assert first != second, \
                "The regions are the same, there is an error in the algorithm!"
            return_tuple = tuple([first, second])
        return return_tuple
    
    def get_trust_control_description(self, n_elements):
        """[summary]
        
        [extended_summary]
        
        Parameters
        ----------
        n_elements : [type]
            [description]
        """
        s = "Initial trust: {}, initial control: {}".format(
            self.parameters["C_share_full_trust"], 
            self.parameters["C_share_full_control_1"])
        return n_elements * [s]
                
    def get_shock_description(self, n_elements):
        """Creates string list for results describing which shocks were used
        
        Parameters
        ----------
        n_elements : int
            The len of the desired list
        
        Returns
        -------
        list
            List that contains `n_elements` times a string that describes
            which shocks were used in the model.
        """
        if self.tshock == "no" and self.cshock == "no":
            s = "No shock"
        else:
            s = "Trust: in t={} to {} Control: in t={} to {}".format(
                self.tshock, self.tshock_value, self.cshock, self.cshock_value)
        return n_elements * [s]
                        
    def report(self, i, initial_round=False, monitor = False):
        params = self.parameters
        if initial_round:
            self.wealth.append(self.record_wealth())
            self.failed_transactions.append(0.0)
            self.rejected_transactions.append(
                sum([a.rejections for a in self.agents]))
            self.realized_transactions.append(0.0)
            self.cooperative_transactions.append(0.0)
            self.exploitations.append(0.0)
            self.defective_transactions.append(0.0)
            self.interactions_with_partners.append(0.0)
            self.interactions_with_strangers.append(0.0)
            self.interactions_by_cooperators.append(0)
            self.interactions_by_defectors.append(0)
            self.pp_interactions_by_cooperators.append(0)
            self.pp_interactions_by_defectors.append(0)
            self.temporal_efficiency.append(0.0)
            self.overall_efficiency.append(0.0)
            self.realized_transactions_share.append(0.0) # TODO? In update this was np.nan, but in original 0.0
            self.failed_transactions_share.append(0.0) # TODO? In update this was np.nan, but in original 0.0

        else:
            self.wealth.append(self.record_wealth())
            max_wealth = params["interactions_per_period"] * \
                params["payoffs"]["a"] * 2.0
            self.temporal_efficiency.append(self.current_t_wealth / max_wealth)
            max_wealth = params["interactions_per_period"] * \
                params["payoffs"]["a"] * 2.0 * (i+1)
            self.overall_efficiency.append(sum(self.wealth[-1]) / max_wealth)
            self.failed_transactions.append(self.failed_transactions_current)
            self.rejected_transactions.append(
                sum([a.rejections for a in self.agents]))
            self.realized_transactions.append(
                self.realized_transactions_current)
            self.realized_transactions_share.append(
                self.realized_transactions_current/params["interactions_per_period"])
            self.failed_transactions_share.append(
                self.failed_transactions_current/params["interactions_per_period"])
            if self.realized_transactions_current > 0:
                self.cooperative_transactions.append(
                    float(self.cooperative_transactions_current)/\
                        self.realized_transactions_current)
                self.exploitations.append(
                    float(self.exploitations_current)/\
                        self.realized_transactions_current)
                self.defective_transactions.append(
                    float(self.defective_transactions_current)/\
                        self.realized_transactions_current)
                self.interactions_with_partners.append(
                    float(self.interactions_with_partners_current)/\
                        self.realized_transactions_current)
                self.interactions_with_strangers.append(
                    float(self.interactions_with_strangers_current)/\
                        self.realized_transactions_current)
            elif self.realized_transactions_current == 0: 
                # TODO? In update the following were np.nan, but in original 0.0
                self.cooperative_transactions.append(0.0)
                self.exploitations.append(0.0)
                self.defective_transactions.append(0.0)
                self.interactions_with_partners.append(0.0)
                self.interactions_with_strangers.append(0.0)
            else:
                exit(1)
            self.interactions_by_cooperators.append(
                self.interactions_by_cooperators_current)
            self.interactions_by_defectors.append(
                self.interactions_by_defectors_current)
            self.pp_interactions_by_cooperators.append(
                0.0 if self.nb_of_ALL_C[-1] == 0 else float(
                    self.interactions_by_cooperators_current)/self.nb_of_ALL_C[-1])
            self.pp_interactions_by_defectors.append(
                0.0 if self.nb_of_ALL_D[-1] == 0 else float(
                    self.interactions_by_defectors_current)/self.nb_of_ALL_D[-1])
            
            if self.realized_transactions_current > 0:
                share_in_coops = float(
                    self.interactions_by_cooperators_current) / \
                        self.realized_transactions_current
                share_in_defects = float(
                    self.interactions_by_defectors_current) / \
                        self.realized_transactions_current
                
                assert share_in_coops + share_in_defects == 2.0, \
                    'Ints by coops and defects not sum to two but to {}'.format(
                        share_in_coops + share_in_defects) 
            else:
                assert self.interactions_by_cooperators_current == 0, \
                    "There should be no transaction at all."
                assert self.interactions_by_defectors_current == 0, \
                    "There should be no transaction at all."

            """Test whether data makes sense"""
            trans_types = self.exploitations[-1] + \
                self.defective_transactions[-1] + self.cooperative_transactions[-1]
            if self.realized_transactions_current == 0:
                pass
            else:
                assert 0.999 < trans_types < 1.001, \
                    "Types of interaction do not sum to one but {}".format(\
                        trans_types)

            assert self.failed_transactions_current + self.realized_transactions_current == \
                   params["interactions_per_period"], \
                       "Error with ints record: failed: {}, realized: {}".format(\
                           self.failed_transactions_current, 
                           self.realized_transactions_current) 
                       
            assert self.cooperative_transactions_current + \
                self.exploitations_current + \
                    self.defective_transactions_current == self.realized_transactions_current, \
                        ("Error with ints recording: realized: "
                         "{}, exploitations: {}, cooperations: {}, defections: {}".format(
                            self.realized_transactions_current, 
                            self.exploitations_current, 
                            self.cooperative_transactions_current, 
                            self.defective_transactions_current))
  
            assert self.interactions_with_partners_current + \
                   self.interactions_with_strangers_current == self.realized_transactions_current, \
                       ("Error reporting ints with partners and strangers: " 
                        "partner: {} stranger: {} total: {}".format(
                           self.interactions_with_partners_current, 
                           self.interactions_with_strangers_current,
                           self.realized_transactions_current)) 

        self.failed_transactions_current = 0
        self.rejected_transactions_current = 0
        self.realized_transactions_current = 0
        self.cooperative_transactions_current = 0
        self.exploitations_current = 0
        self.defective_transactions_current = 0
        for ag in self.agents:
            ag.reset_rejections()
            ag.reset_payoff_hist() 
        self.interactions_with_partners_current = 0
        self.interactions_with_strangers_current = 0
        self.interactions_by_defectors_current = 0
        self.interactions_by_cooperators_current = 0
        self.current_t_wealth = 0.0
        
    def get_agent(self, id):
        """Gets an agent with given id
                
        Parameters
        ----------
        id : int
            The id of the desired agent
        
        Returns
        -------
        agent.Agent
            The agent.Agent with the desired `id`.
        """
        assert isinstance(id, int), \
            "Id must be given as integer, not {}.".format(type(id))
            
        agent_to_return = next((x for x in self.agents if x.ident == id), None)
        
        assert isinstance(agent_to_return, Agent), \
            "The returned agent must be of type agents, not {}".format(
                type(agent_to_return))
            
        assert agent_to_return.ident == id, \
            "Something went wrong with agent selection!"
            
        return agent_to_return
    
    def record_wealth(self):
        """Stores the wealth of every agent in a list.
        """
        list_of_wealth = []
        for ag in self.agents:
            list_of_wealth.append(sum(ag.wealth))
        return list_of_wealth
    
    def select_hawaladar(self, region):
        """Selects randomly an hawala from the given region.
                
        Parameters
        ----------
        region : int
            The region from which an agent is to be chosen
        
        Returns
        -------
        [type]
            [description]
        """
        assert isinstance(region, (int, np.int64)), \
            "Region must be int, not {}".format(type(region))
        selected_hawaladar = np.random.choice(self.places[region])
        assert isinstance(selected_hawaladar, Agent), \
            'The selected hawaladar is not type agent but {}'.format(
                type(selected_hawaladar))
        return selected_hawaladar

    def play(self, id_1, id_2):
        """Two hawaladars play a PD against each other
        # TODO : rename id_1 to ag_1
        
        1. Get payoff matrice from parameters and collect strategies of agents
        2. If relevant, change strategies according to mistake probability
        3. Compute the payoffs as (:math:`A_i` is payoff matrix, 
        :math:`s_i` strategy):
        # TODO! Check whether payoff computation is correct
        .. math::
        
            \Pi_1 = [s_1 \times A_1] \times s_2\\
            \Pi_2 = [s_1 \times A_2] \times s_2    
        
        4. Documentation of results
        5. Actual distribution of payoffs
        
        Parameters
        ----------
        id_1, id_2 : Agent
            The player instances that play against each other.
        
        Returns
        -------
        Nothing
        """
        # 1. Preparation and collection of strategies
        assert id_1.ident != id_2.ident, 'A player must not play against herself!'
        params = self.parameters
        payoffs = params['payoffs']
        a, b, c, d = float(payoffs['a']), float(payoffs['b']), \
            float(payoffs['c']), float(payoffs['d'])
        player_one_payoff_matrix = np.array(((a, d), (b, c)))
        player_two_payoff_matrix = np.array(((a, b), (d, c)))
        strategy_player_1 = id_1.get_strategy()
        strategy_player_2 = id_2.get_strategy()
        
        # 2. Mistakes
        if self.mistake_prob > 0.0:
            if np.random.random() < self.mistake_prob:
                s1_old = strategy_player_1
                strategy_player_1 = np.array((1, 0)) if np.array_equal(
                    s1_old, np.array((0, 1))) else np.array((0, 1))
            if np.random.random() < self.mistake_prob:
                s2_old = strategy_player_2
                strategy_player_2 = np.array((1, 0)) if np.array_equal(
                    s2_old, np.array((0, 1))) else np.array((0, 1))
                # TODO: Check that for all statistics that may use of that!
        
        # 3. Computation of payoffs
        payoff_id1 = np.dot(np.dot(strategy_player_1, 
                                   player_one_payoff_matrix), 
                            strategy_player_2)
        payoff_id2 = np.dot(np.dot(strategy_player_1, 
                                   player_two_payoff_matrix), 
                            strategy_player_2)
        
        # 4. Documentation of results
        self.realized_transactions_current += 1
        self.current_t_wealth += (payoff_id1 + payoff_id2)
        if np.array_equal(id_1.get_strategy(), np.array((1, 0))):
            self.interactions_by_cooperators_current += 1
        if np.array_equal(id_2.get_strategy(), np.array((1, 0))):
            self.interactions_by_cooperators_current += 1
        if np.array_equal(id_1.get_strategy(), np.array((0, 1))):
            self.interactions_by_defectors_current += 1
        if np.array_equal(id_2.get_strategy(), np.array((0, 1))):
            self.interactions_by_defectors_current += 1
        check_plausib = True
        if (np.array_equal(strategy_player_1, np.array((1, 0)))) and \
            (np.array_equal(strategy_player_2, np.array((1, 0)))):
            self.cooperative_transactions_current += 1
            if check_plausib:
                assert payoff_id2 == a, \
                    'Something wrong with payoff calculation.'
        if ((np.array_equal(strategy_player_1, np.array((1, 0)))) and \
            (np.array_equal(strategy_player_2, np.array((0, 1))))) or \
                ((np.array_equal(strategy_player_1, np.array((0, 1)))) and \
                    (np.array_equal(strategy_player_2, np.array((1, 0))))):
            self.exploitations_current += 1
            if check_plausib:
                assert payoff_id2 == b or payoff_id2 == d, \
                    'Something wrong with payoff calculation'
                assert payoff_id2 != payoff_id1, \
                    'Something wrong with payoff calculation'
        if (np.array_equal(strategy_player_1, np.array((0, 1)))) and \
            (np.array_equal(strategy_player_2, np.array((0, 1)))):
            self.defective_transactions_current += 1
            if check_plausib:
                assert payoff_id2 == c, \
                    'Something wrong with payoff calculation'

        # 5. Actual distribution of payoffs
        id_1.receive_payoff(payoff_id1, id_2, params)
        id_2.receive_payoff(payoff_id2, id_1, params)
    
    def record_strats(self):
        """Record strategies currently used by agents
        
        Records number of agents using a particular strategy in a time step.
        """
        strats = [a.strategy for a in self.agents]
        n_all_c = strats.count('ALL_C')
        n_all_d = strats.count('ALL_D')
        assert n_all_c + n_all_d==len(self.agents),\
            'Sum of recorded strats is {} but there are {} agents!'.format(
                n_all_c+n_all_d, len(self.agents))
        self.nb_of_ALL_C.append(n_all_c)
        self.nb_of_ALL_D.append(n_all_d)
        self.share_of_ALL_C.append(float(n_all_c) / len(self.agents))
        self.share_of_ALL_D.append(float(n_all_d) / len(self.agents))
        del strats
        
    def rank_agents(self):
        """Ranks agents according to success and records best/worst strategies

        1. Rank agents. There are currently 4 different ways to do so:
            a) `total_wealth`: ranks agents according to their total wealth 
                (`get_total_wealth`)
            b) `last_wealth_abs`: ranks agents according to the absolute wealth 
                received in the previous period
            c) `last_wealth_av`: ranks agents according to the average wealth 
                received in the last `x` periods (default: `x=5`)
            d) `last_wealth_sum`: ranks agents according to the absolute wealth 
                received in the last `x` periods (parameter `last_wealth_per`)
        2. Choosing the best and worst agents and determine their strategy 
            distribution

        Returns
        -------
        dict
            Contains entries for the best and worst agents and the strategy
            distribution for the two groups.
            
        Raises
        ------
        InternalError
            If the wrong selector is specified in the parameters.
        """
        # 1. Rank agents
        selector = self.parameters['selector']
        select_percent = float(self.parameters['selection_percentage'])
        agent_list = self.agents
        np.random.shuffle(agent_list)
        
        if selector == 'total_wealth':
            ranked_agents = sorted(agent_list, 
                                   key=methodcaller('get_total_wealth'), 
                                   reverse=True)
        elif selector == 'last_wealth_abs':
            ranked_agents = sorted(agent_list, 
                                   key=methodcaller('get_last_wealth_abs', 1), 
                                   reverse=True)
        elif selector == 'last_wealth_av':
            ranked_agents = sorted(agent_list, 
                                   key=methodcaller('get_last_wealth_av', 5), 
                                   reverse=True)
        elif selector == 'last_wealth_sum':
            rounds_considered = self.parameters["last_wealth_per"]
            ranked_agents = sorted(agent_list, 
                                   key=methodcaller('get_last_wealth_abs',
                                                    rounds_considered), 
                                   reverse=True)
        else:
            raise InternalError('Wrong selector: {}'.format(selector))
            
        # 2. Determine top and low agents
        top_agents = ranked_agents[:int(select_percent*len(ranked_agents))]  
        # assumes the rich agents are at beginning
        low_agents = ranked_agents[-int(select_percent*len(ranked_agents)):]  
        # assuming poor agents are at the end
        
        # 3. Compute strategy distribution for top and low agents
        top_agents_strats = [a.strategy for a in top_agents]

        share_c_top = \
            top_agents_strats.count('ALL_C') / float(len(top_agents_strats))
        share_d_top = \
            top_agents_strats.count('ALL_D') / float(len(top_agents_strats))
        assert share_c_top + share_d_top == 1.0, \
            'Strat shares sum to {}'.format(share_c_top+share_d_top)
            
        low_agents_strats = [a.strategy for a in low_agents]
        share_c_low = \
            low_agents_strats.count('ALL_C') / float(len(low_agents_strats))
        share_d_low = \
            low_agents_strats.count('ALL_D') / float(len(low_agents_strats))
        assert share_c_low + share_d_low == 1.0, \
            'Strat shares sum to {}'.format(share_c_low+share_d_low)
        
        # 4. Return relevant information
        return_dict = dict()
        return_dict["top_agents"] = top_agents
        return_dict["low_agents"] = low_agents
        return_dict["share_c_top"] = share_c_top
        return_dict["share_d_top"] = share_d_top
        return_dict["share_c_low"] = share_c_low
        return_dict["share_d_low"] = share_d_low
        
        return(return_dict)
    
    def selection_strats(self):
        """Selection and replication of the strategies
        
        The share of agents that are used as reference group at the top and 
        that change their behavior at the bottom are given by the parameter 
        `selection_percentage`. 
        
        The function then works as follows:
        1. Rank agents. There are currently 4 different ways to do so:
            a) `total_wealth`: ranks agents according to their total wealth 
                (`get_total_wealth`)
            b) `last_wealth_abs`: ranks agents according to the absolute wealth 
                received in the previous period
            c) `last_wealth_av`: ranks agents according to the average wealth 
                received in the last `x` periods (default: `x=5`)
            d) `last_wealth_sum`: ranks agents according to the absolute wealth 
                received in the last `x` periods (parameter `last_wealth_per`)
        2. Choosing the best and worst agents and determine their strategy 
            distribution
        
        Note: these two steps are made via the function `self.rank_agents()`
        
        3. Adapt strategies of the worst agents. There are two options, which 
        are selected by the parameter `adaptive_strategy_selection`:
            a) `random`: the new strategies are chosen by assigning every
                strategy the same probability to be chosen.
            b) `replication_top_ten`: the new strategies are drawn from the 
                distribution of strategies of the top agents.
        
        Returns
        -------
        Nothing
        
        Raises
        ------
        InternalError
            If no adequate selection or adaption mechanism has been specified.
        """
        # 1. Rank agents
        ranking_result = self.rank_agents()
        top_agents = ranking_result["top_agents"]
        low_agents = ranking_result["low_agents"]
        share_c_top = ranking_result["share_c_top"]
        share_d_top = ranking_result["share_d_top"]
        share_c_low = ranking_result["share_c_low"]
        share_d_low = ranking_result["share_d_low"]
        
        # 2. Update information about top and low agents
        self.share_ALL_C_top.append(share_c_top)
        self.share_ALL_D_top.append(share_d_top)
        self.share_ALL_C_low.append(share_c_low)
        self.share_ALL_D_low.append(share_d_low)
            
        # 3. Update strategies
        update_mechanism = self.parameters['adaptive_strategy_selection']
        if update_mechanism == 'random':
            for ag in low_agents:
                newstrat = np.random.choice(['ALL_C', 'ALL_D'])
                ag.change_strat(
                    newstrat, 
                    parameterfile=self.parameters, 
                    trust_shock=self.tshock_has_taken_place, 
                    control_shock=self.cshock_has_taken_place)

        elif update_mechanism == 'replication_top_ten':
            top_agents_strats = [a.strategy for a in top_agents]
            for ag in low_agents:
                newstrat = np.random.choice(top_agents_strats)
                ag.change_strat(
                    newstrat, 
                    parameterfile=self.parameters, 
                    trust_shock=self.tshock_has_taken_place, 
                    control_shock=self.cshock_has_taken_place)
        else:
            raise InternalError(
                'Wrong adaption mechanism: {}'.format(update_mechanism))

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

class InternalError(Error):
    """Exception raised internal code inconsistencies.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
    