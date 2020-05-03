import sys; sys.path.insert(0, './python')
import os
import random
import agent
import model
import matplotlib.pyplot as plt
import numpy as np
from agent import Agent
file_marker = "[" + str(os.path.basename(__file__)) + "]: "


class PopulationGenerator:
    """
    This class is used only to initialize the population of hawaladars.
    """
    def __init__(self, parameter_file, model_instance):
        """Initializes a population of agents

        Parameters
        ----------
        parameter_file : dict
            The parameters of the model
        model_instance : model.Model
            The model instance for which the population should be created
        """
        assert type(parameter_file) == dict, \
            "Parameters given in the wrong format!"
        self.params = parameter_file
        assert isinstance(model_instance, model.Model), \
            "Given model instance not of type model {}".format(
                type(model_instance))
        self.model_instance = model_instance
        self.agents = []
        self.places = dict()
        
        self.initial_share_ALL_C = model_instance.initial_share_ALL_C
        self.initial_share_ALL_D = model_instance.initial_share_ALL_D
        self.n_hawaladars = self.params['number_of_hawaladars']
        
        self.strategy_distribution = \
            int(self.initial_share_ALL_C*self.n_hawaladars)*['ALL_C'] + \
                    int(self.initial_share_ALL_D*self.n_hawaladars)*['ALL_D']
        if len(self.strategy_distribution) < self.n_hawaladars:
            self.strategy_distribution.append(['ALL_C'])
        random.shuffle(self.strategy_distribution)

        assert len(self.strategy_distribution) == self.n_hawaladars, \
            'Incorrect strategy dist. Len strats: {} len hawal: {}'.format(
                len(self.strategy_distribution), self.n_hawaladars)
            
        self.init_population()


    def chunk(self, _list, num_parts):
        """
        Takes a list and splits it into num parts.
        """
        avg = len(_list) / float(num_parts)
        out = []
        last = 0.0
        while last < len(_list):
            out.append(_list[int(last):int(last + avg)])
            last += avg
        return out

    def make_dict(self, list_of_agents, list_of_places):
        """
        Takes a list of agents and a list of places and returns
        a dictionary with the agents as keys and their location as
        their corresponding value.
        """

        allocation_1 = self.chunk(list_of_agents, len(list_of_places))
        allocation_2 = [zip(allocation_1[i], 
                            len(allocation_1[i])*[list_of_places[i]])
                        for i in range(len(list_of_places))]
        allocation_final = dict()
        for i in range(len(allocation_2)):
            for k, v in allocation_2[i]:
                allocation_final[k] = v
        return allocation_final
    
    @staticmethod
    def is_prob_float(number_to_test):
        """Test whether the nb is a float between 0 and 1
        
        Parameters
        ----------
        number_to_test : float
            A number to test

        Returns
        -------
        bool
            True if `number_to_test` is a float and in (0,1). False otherwise.
        """
        if (0.0 <= number_to_test <= 1.0) and isinstance(number_to_test, float):
            test_result = True
        else:
            test_result = False
        return test_result

    def init_population(self):
        """
        Initializes the population.
        Returns a tuple of agents and a dictionary for the places.
        """
        agents = []
        number_of_agents = self.params['number_of_hawaladars']
        number_of_places = self.params['number_of_places']
        list_of_agents = list(range(0, number_of_agents))
        random.shuffle(list_of_agents)
        list_of_places = list(range(0, number_of_places))
        allocation = self.make_dict(list_of_agents, list_of_places)
        
        c_trust = self.params["C_trust"]
        assert self.is_prob_float(c_trust), \
            "C_trust wrong({}, type: {})".format(
                c_trust, type(c_trust))
        
        d_trust = self.params["D_trust"]
        assert self.is_prob_float(d_trust), \
            "D_trust wrong({}, type: {})".format(
                d_trust, type(d_trust))
        
        c_control_1 = self.params["C_control_1"]
        assert self.is_prob_float(c_control_1), \
            "C_control_1 wrong({}, type: {})".format(
                c_control_1, type(c_control_1))
        
        c_control_2 = self.params["C_control_2"]
        assert self.is_prob_float(c_control_2), \
            "C_control_2 wrong({}, type: {})".format(
                c_control_2, type(c_control_2))
        
        d_control_1 = self.params["D_control_1"]
        assert self.is_prob_float(d_control_1), \
            "D_control_1 wrong({}, type: {})".format(
                d_control_1, type(d_control_1))
        
        d_control_2 = self.params["D_control_2"]
        assert self.is_prob_float(d_control_2), \
            "D_control_2 wrong({}, type: {})".format(
                d_control_2, type(d_control_2))

        cooperators = \
            int(self.initial_share_ALL_C*self.params['number_of_hawaladars'])
        defectors = self.params['number_of_hawaladars'] - cooperators
        assert cooperators + defectors == self.params['number_of_hawaladars'], \
            "Cooperators & Defectors dont sum up. Should be {} but is {}.".format(
                self.params['number_of_hawaladars'], sum(cooperators, defectors))

        trusting_coops = \
            int(self.params['C_share_full_trust'] * cooperators)*[c_trust]
        non_trusting_coops = (cooperators - len(trusting_coops))*[0.0]
        cooperators_trusts = trusting_coops + non_trusting_coops
        random.shuffle(cooperators_trusts)
        assert len(cooperators_trusts) == cooperators, \
            "Len of trusting cooperators is {} but should be {}.".format(\
                len(cooperators_trusts), cooperators)

        controlling_coops_1 = \
            int(self.params['C_share_full_control_1'] * cooperators) * [c_control_1]
        non_controlling_coops_1 = \
            (cooperators - len(controlling_coops_1)) * [0.0]
        cooperators_controls_1 = controlling_coops_1 + non_controlling_coops_1
        random.shuffle(cooperators_controls_1)
        assert len(cooperators_controls_1) == cooperators, \
            "Len of trusting cooperators is {} but should be {}.".format(
                len(cooperators_controls_1), cooperators)

        controlling_coops_2 = \
            int(self.params['C_share_full_control_2'] * cooperators) * [c_control_2]
        non_controlling_coops_2 = (cooperators - len(controlling_coops_2)) * [0.0]
        cooperators_controls_2 = controlling_coops_2 + non_controlling_coops_2
        random.shuffle(cooperators_controls_2)
        assert len(cooperators_controls_2) == cooperators, \
            "Len of trusting cooperators is {} but should be {}.".format(
                len(cooperators_controls_2), cooperators)

        nans_cooperators = cooperators*[np.NaN]  # List iteration will start at index cooperators

        controlling_defects_1 = \
            int(self.params['D_share_full_control_1'] * defectors) * [d_control_1]
        non_controlling_defects_1 = (defectors - len(controlling_defects_1)) * [0.0]
        defectors_controls_1 = controlling_defects_1 + non_controlling_defects_1
        random.shuffle(defectors_controls_1)
        assert len(defectors_controls_1) == defectors, \
            "Len of trusting defectors is {} but should be {}.".format(
                len(defectors_controls_1), defectors)
        defectors_controls_1 = nans_cooperators + defectors_controls_1

        controlling_defects_2 = \
            int(self.params['D_share_full_control_2'] * defectors) * [d_control_2]
        non_controlling_defects_2 = \
            (defectors - len(controlling_defects_2)) * [0.0]
        defectors_controls_2 = controlling_defects_2 + non_controlling_defects_2
        random.shuffle(defectors_controls_2)
        assert len(defectors_controls_2) == defectors, \
            "Len of trusting defectors is {} but should be {}.".format(
                len(defectors_controls_2), defectors)
        defectors_controls_2 = nans_cooperators + defectors_controls_2

        interval_defectors = defectors * [self.params['D_interval']]  # The interval between defections

        assert len(interval_defectors) == defectors, \
            "Len of trusting cooperators is {} but should be {}.".format(
                len(cooperators_controls_2), defectors)
        interval_defectors = nans_cooperators + interval_defectors


        for i in range(0, cooperators):
            agents.append(
                Agent(
                    list_of_agents[i], 
                    self.model_instance, 
                    allocation[i],
                    cooperators_trusts[i], 
                    cooperators_controls_1[i],
                    cooperators_controls_2[i], 
                    np.NaN, 
                    strategy='ALL_C',
                    partner_selection=self.params["partner_selection"]))

        for i in range(cooperators, cooperators+defectors):
            agents.append(
                Agent(
                    list_of_agents[i], 
                    self.model_instance, 
                    allocation[i],
                    np.NaN, 
                    defectors_controls_1[i],
                    defectors_controls_2[i], 
                    interval_defectors[i], 
                    strategy='ALL_D',
                    partner_selection=self.params["partner_selection"]))

        places = dict()
        for key in range(0, number_of_places):
            places[key] = []
        print('Nb of regions: ', number_of_places)
        assert len(places.keys()) == number_of_places, \
            'Places dict is wrong: nb of key wrong!'
        assert isinstance(places[0], list), \
            'Places dict is wrong: Values are no lists'
            
        for ag in agents:
            assert isinstance(ag, Agent), \
                'Hier ist was mit dem Typ der Agenten faul!'
            assert isinstance(ag.place, int), \
                'Region not given by int.'
            places[ag.place].append(ag)
        assert isinstance(places[0][0], agent.Agent), \
            'Places dict does not include agents as values.'
        self.agents = agents
        self.places = places

    def check_assumptions(self):
        assert len(self.agents) == self.params['number_of_hawaladars'], \
            'Length of agent list smaller than nb of hawaladars!'
        assert isinstance(self.agents[0], agent.Agent) == 1, \
            'Agents in the list are not instantiations of the agent class!'
        assert len(self.places.keys()) == self.params['number_of_places'], \
            'Number of places in places dict too small!'
        assert len(self.places.values()) == self.params['number_of_places'], \
            'Places dict is not one-to-one!'
        for i in range(0, len(self.places.keys())):
            assert isinstance(self.places[i], list), \
                'The values of the places dict are not lists'
            assert len(self.places[i]) == len(self.agents)/len(self.places), \
                'Agents seem to be allocated wrongly'
        for i in self.places.keys():
            print('length of region ' + str(i) + ' is ' + \
                str(len(self.places[i])) + '  ' + str(self.places[i]))
        for i in self.agents:
            assert self.agents.index(i) == i.get_ident(), \
                'Agents id not equal to her place in the list!'
        for i in self.agents:
            assert isinstance(i, agent.Agent), \
                'ERROR: Loop for checking agents allocation flawed!'
            is_in_places = False
            for a in range(0, len(self.places.values())):
                if i in self.get_places().values()[a]:
                    is_in_places = True
            assert is_in_places == True, \
                'An agent has not been allocated to a region!'
        print("Successfully checked allocation, indexes and ids of the agents")

    def get_agents(self):
        return self.agents

    def get_places(self):
        return self.places
