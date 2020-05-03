import sys; sys.path.insert(0, './python')
import os
import numpy as np
file_marker = "[" + str(os.path.basename(__file__)) + "]: "

class Strategy:
    """
    This class includes all possible strategies.
    Each round, the number of users for strategies gets recorded.
    The model.py file should ensure that every round this number gets recorded.
    """

    def __init__(self, model):
        self.associated_model = model
        self.number_of_timesteps = model.parameters['number_of_timesteps']
        self.list_of_strategies = ['ALL_C', 'ALL_D']
        self.counter_ALL_C = 0
        self.hist_ALL_C = self.number_of_timesteps*[np.nan]
        self.counter_ALL_D = 0
        self.hist_ALL_D = self.number_of_timesteps*[np.nan]

    def give_strat(self, name):
        """
        Parameters
        ----------
        name: str
            Name of the desired strategy

        Returns
        -------
        np.array
            Returns the array representation of 
            cooperation (1,0) or defection (0,1)
        """
        if name == 'ALL_C':
            # The agent always cooperates
            self.counter_ALL_C += 1
            return np.array((1, 0))
        if name == 'ALL_D':
            # The agent always defects
            self.counter_ALL_D += 1
            return np.array((0, 1))
        else:
            assert 1 > 2, \
                'There is a strategy no defined in the strategy object!'

    def reset_counters(self):
        self.counter_ALL_C = 0
        self.counter_ALL_D = 0

    def record_counters(self, timestep):
        self.hist_ALL_C[timestep] = self.counter_ALL_C
        self.hist_ALL_D[timestep] = self.counter_ALL_D

    def get_counter_history(self):
        """
        Parameters
        ----------
        None

        Returns
        -------
        tuple of lists
            The tuple contains two lists that both contain the nb of 
            cooperators and defectors at each time step.
        """
        assert np.nan not in self.hist_ALL_C, \
            'Not all strategy distributions recorded for ALL-C.'
        assert np.nan not in self.hist_ALL_D, \
            'Not all strategy distributions recorded for ALL-D.'
        return_hist = [self.hist_ALL_C, self.hist_ALL_D]
        return tuple(return_hist)
