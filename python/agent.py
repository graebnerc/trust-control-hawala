import sys; sys.path.insert(0, './python')
import os
import random
import numpy as np

file_marker = "[" + str(os.path.basename(__file__)) + "]: "


class Agent:
    """
    This is our hawala agent
    """
    def __init__(self, identification, model, place, 
                 trust, control_1, control_2, interval, strategy='ALL_C',  
                 partner_selection=True):

        self.partner_selection = partner_selection  # True or false; if true, H. choose partners acc to reputation
        self.ident = identification  # A number for identification purposes
        self.model = model  # The model instance to which the agent belongs
        self.place = place  # The region the agent is located in
        self.has_played = False  # Initialized as false; indicates whether agent was active in a round
        self.connections = dict()  # The relationships the agent has formed; init as empty
        # This is a dictionary of lists: keys are the regions, values a list with known hawaladars in this region
        self.reputations = dict()  # If payoff obtained from interaction is greater than c, this payoff is stored
        self.gossip = dict()  # Stores information on gossip, i.e. negative reputation from others
        self.wealth = [0.0]  # Stores the wealth of the agent; in round 0 wealth equals 0.0
        self.strategy = strategy  # The strategy of the agent: either cooperating or defecting
        self.strategy_object = self.model.strategy_class  # The associated strat object
        self.has_changed_strat = False  # Becomes True once the agent has changed his strategy
        self.rejections = 0  # The number of interactions the agent has rejected
        self.payoff_history = []  # The payoffs from past interactions of the agent
        # Newly added stuff
        self.trust = trust  # Probability to interact with strangers; relevant only for cooperators
        self.interval = interval # Period between two defections; relevant only for defectors TODO Implement interval
        self.control_1 = control_1  # Probability to reject those who have cheated on him in the past TODO Implement control 1
        self.control_2 = control_2  # Probability to reject those who have cheated on her partners in the past TODO Implement control 2

    def __repr__(self):
        return '['+'A_'+str(self.ident)+'_R_'+str(self.place)+']'
    
    def get_total_wealth(self):
        return sum(self.wealth)
    
    def get_last_wealth_abs(self, n):
        """Get sum of wealth accumulated in the previous `n` rounds
                
        Parameters
        ----------
        n : int
            Nb of past rounds for which wealth is to be summed up
        
        Returns
        -------
        float
            Sum of wealth from the previous `n` rounds
        """
        return sum(self.wealth[-n:])

    def get_last_wealth_av(self, n):
        """Get average of wealth accumulated in the previous `n` rounds
                
        Parameters
        ----------
        n : int
            Nb of past rounds for which wealth is to be summed up
        
        Returns
        -------
        float
            Average of wealth accumulated in the previous `n` rounds
        """
        return float(sum(self.wealth[-n:]))/len(self.wealth[-n:])

    def reset_payoff_hist(self):
        """Reset the payoff history of the current round
        
        Gets called after all interactions of one time step have taken place
        and resests the payoff history. Does not affect the wealth history, 
        which is never reseted.
        """
        self.payoff_history = []
    
    def reset_rejections(self):
        self.rejections = 0
    
    def shock(self, kind_of_shock, shock_value):
        """Sets a property of the agent exogeneously to new value
        
        [extended_summary]
        
        Parameters
        ----------
        kind_of_shock : str
            [description]
        shock_value : float
            [description]
        
        Raises
        ------
        AssertionError
            If `kind_of_shock` has no adequate value.
        """
        assert isinstance(kind_of_shock, str), \
            "kind of shock not given as str but {}".format(type(shock_value))
        assert isinstance(shock_value, float), \
            "shock variable not float but {}".format(type(shock_value))
        assert 0 <= shock_value <= 1.0, \
            "shock value not in [0, 1]: {}".format(shock_value)
            
        if kind_of_shock == "trust":
            if self.strategy == "ALL_D":
                pass
            else:
                self.trust = float(shock_value)
        elif kind_of_shock == "control":
            if self.strategy == "ALL_D":
                pass
            else:
                self.control_2 = float(shock_value)
                self.control_1 = float(shock_value)
        elif kind_of_shock == "control_1":
            if self.strategy == "ALL_D":
                pass
            else:
                self.control_1 = float(shock_value)
        elif kind_of_shock == "control_2":
            if self.strategy == "ALL_D":
                pass
            else:
                self.control_2 = float(shock_value)
        else:
            # TODO Write error classes
            raise AssertionError("Wrong input: %s") % str(kind_of_shock)
    
    def check_connection(self, region):
        """Checks whether the agent knows a hawala in `region`.
        
        If partner_selection is turned off, then agents will never prefer 
        partners with a good interaction history. In this case the function  
        returns false, and function `get_stranger()` is always used for partner 
        selection.
        
        Parameters
        -----------
        region: int
            An integer referring to one of the regions in the model
        
        Returns
        -------
        True/False
            True if partner selection is activated and the agent has a 
            connection to one or more agents in the region; False otherwise.
        """
        assert isinstance(region, (int, np.int64)), \
            "Region should be given as int, not {}".format(type(region))
        if self.partner_selection == 0:
            return_value = False
        elif region in self.connections.keys():
            return_value = True
        else:
            return_value = False
        return return_value
    
    def get_partner(self, region):
        """Gets a known partner from `region`
        
        Is called only  when `check_connection()` was True.
        
        Parameter
        ---------
        region: int
            One of the regions of the model.

        Returns
        -------
        partner: Agent or False
            The agent the current agent is goint to interact with. If the 
            chosen agent does not agree to the interaction, the function 
            returns False.        
        """
        partner_id = random.choice(tuple(self.connections[region]))
        assert isinstance(partner_id, int), \
            "Partner id in connections should be given as int, not {}".format(
                type(partner_id))
        partner = self.model.get_agent(partner_id)
        
        if partner.check_acceptance(self.ident, self.model.parameters) is True:
            assert self.ident != partner.ident, \
                'Partner is equal to the agent herself!'
            return_object = partner
            assert isinstance(partner, Agent), \
                "Chosen partner not Agents but {}".format(type(partner))
        else:
            return_object = False
        return return_object

    def check_acceptance(self, id_partner, parameter_file):
            """Checks whether an agent agrees to a proposed interaction
            
            If the agent is chosen by another hawala as a potential partner,
            the agent checks whether she accepts to trade with the other hawaladar,
            based on potential gossip about it.
            Then one adds a 0.0 to the gossip list, otherwise gossip will last forever.
            If 0.0 is added this captures the idea of the time lag for the gossip list.
            
            If there has been a positive interaction with the partner in the past,
            the agent agrees to the interaction.
            
            If there is gossip about the parner, then the interaction will be 
            refused with the probability as specified in the parameters. If not,
            the gossip gets ignored. In this case, a defector then always enters
            the interaction, for a cooperator it depends on the personal trust.
            # TODO does this make sense?
            
            If there is no information about the partner, defectors will always
            agree to the interaction, for cooperators it depends on their trust.
            
            Note: `np.random.binomial(1, self.trust)` simulates one trial of the
            Binomial distribution with success probability of `self.trust`.
            
            Parameters
            ----------
            id_partner: int
                The identification if for the potential partner
            parameter_file: dict
                The parameters associated with the model run

            Returns
            -------
            bool
                True if the agent accepts the interaction; False otherwise
            """
            assert isinstance(id_partner, int), \
                "Partner id not given as int but {}".format(type(id_partner))
                
            lag = parameter_file['time_lag_gossip']
            
            # If there has been a positive interaction with agent in the past:
            if (id_partner in self.reputations.keys()) and \
                (self.reputations[id_partner][-1] >= parameter_file['payoffs']['a']):
                return_value = True
                
            # If there is some gossip about the potential partner agent available 
            # since the agent of a partner had a negative interaction in the past:
            elif id_partner in self.gossip.keys():
                rdn = random.random()
                if (rdn > max(self.gossip[id_partner][lag:])):
                    # Check whether rdn higher than rejection prob
                    self.gossip[id_partner].append(0.0)
                    
                    if self.strategy == "ALL_D":
                        return_value = True
                    else:
                        # For cooperators it depends on their trust whether they
                        # enter this interaction
                        if np.random.binomial(1, self.trust) == 1.0:
                            return_value = True
                        else:
                            return_value = False
                else:
                    self.gossip[id_partner].append(0.0)
                    self.rejections += 1
                    return_value =  False
                    
            # If there is no information on this agent:
            elif (id_partner not in self.gossip.keys()) and \
                (id_partner not in self.reputations.keys()):
                # Defectors always interact in this case
                if self.strategy == "ALL_D":  
                    return_value =  True
                # For cooperators it depends on their trust:
                else:
                    if np.random.binomial(1, self.trust) == 1.0:
                        return_value =  True
                    else:
                        return_value =  False
                        
            # The following case should not happen
            else:
                print("max gossip: ", max(self.gossip[id_partner][lag:]), "\n",
                    "Partner in gossip list: ", 
                    (id_partner in self.gossip.keys()),
                    "Partner in reputation list: ", 
                    (id_partner in self.reputations.keys())
                    )
                if id_partner in self.reputations.keys():
                    print("Previous value in reputation list: ", 
                        self.reputations[id_partner][-1])
                raise InternalError("Error in check_acceptance function!")
            
            return return_value
    
    def get_stranger(self, agents_in_region, parameter_file, region):
        """Find an interaction partner that is so far not known to agent
        
        Gets a random hawala from a region. Is called when check_connection 
        was false.
        
        1. If the agent is cooperator, check whether he has sufficient trust
        2. Go through all agents in this region and test...
        2.1. If it is the agent herself, then pass
        2.2. Elif the agent has reputation and this is above payoff c 
            (i.e. the interaction has been positive); in this case add the
            agent to the possibles list (but throw error if `partner_selection`
            was True).
        2.3. Elif there is gossip about the agent, the place her on list
            of possibles with probability 
            `random.random() > max(self.gossip[ag.ident][lag:])`
            # tODO! check this
        2.4. Elif the agent is neither in reputation nor in gossip list, then
            add her to list of possibles
        3. Remove all agents from list of possibles that do not want to
            interact
        4. Choose one of the remaining agents randomly
        
        Parameters
        ----------
        agents_in_region: list of Agents
            A list with all the agents in the region
        parameter_file: open json file
            The associated parameter file for the model run.

        Returns
        -------
        Agent
            The agent that has been chosen from the region.
            Return False if no agent could be chosen.
        
        # TODO: The lag parameter might require revision:
        it does not capture the actual time lag, but lag for interactions.
        """
        assert isinstance(agents_in_region, list), \
            'Method get_stranger requires list of agents as input!'
        assert isinstance(agents_in_region[0], Agent), \
            'Method get_stranger requires list of agents as input!'
        assert self.check_connection(region) == False, \
            "get_stranger method should not have been called!"
            
        lag = parameter_file['time_lag_gossip']
        possibles = []
        if (self.strategy=="ALL_C") and (np.random.binomial(1, self.trust)==0.0):
            """This agent has not enough trust to engage in an interaction."""
            chosen = False
        else:
            for ag in agents_in_region:
                if ag.ident == self.ident:
                    pass
                elif (ag.ident in self.reputations.keys()) and \
                                self.reputations[ag.ident][-1] > \
                                    parameter_file['payoffs']['c']: 
                    # If there has been positive interaction in past
                    assert self.partner_selection is False, \
                        "Agent should have been in connection list"
                    possibles.append(ag)
                elif ag.ident in self.gossip.keys():  
                    # If there is gossip information about this agent
                    if random.random() > max(self.gossip[ag.ident][lag:]):
                        possibles.append(ag)
                    else:  
                        # If there is no information about the agent
                        self.rejections += 1
                elif ag.ident not in self.gossip.keys() and \
                    ag.ident not in self.reputations.keys():
                    possibles.append(ag)
                else:
                    raise InternalError("No serious if condition taken!")

            rejectors = []
            for ags in possibles:
                if ags.check_acceptance(self.ident, parameter_file) == 1:
                    pass
                else:
                    rejectors.append(ags)
                    self.rejections += 1
            possibles = [x for x in possibles if x not in rejectors]
            if possibles == []:
                chosen = False
            else:
                chosen = random.choice(possibles)
                assert isinstance(chosen, Agent), \
                    'The chosen stranger is not of type agent: {}'.format(chosen)
        return chosen
    
    def get_strategy(self):
        """
        Parameters
        ----------
        None

        Returns
        -------
        np.array
            An array indicating the action of the agent: 
            (0,1) corresponds to defection, (1,0) to cooperation
        """
        requested_strategy = str(self.strategy)  # 'ALL_C' or 'ALL_D' # TODO Adjust to new classification
        strategy_object = self.strategy_object  
        strategy_array = strategy_object.give_strat(requested_strategy)
        return strategy_array
    
    def receive_payoff(self, payoff, opponent, parameter_file):
        """Allocate payoffs and update gossip and partner dictionaries
        
        
        Parameters
        ----------
        payoff: float
            The payoff received by the interaction

        opponent: Agent
            The interaction partner of the agent

        parameter_file: open json file
            The parameter file associated with the simulation run

        Returns
        -------
        None

        Functioning
        -----------
        After having played against another player, the agent receives her 
        payoff.

        If she got exploited, she adds the agent to her gossip dictionary and 
        tells all the other agents she had a positive previous interaction 
        about the exploitation.
        If she did not exploited, the region of the opponent gets added to the 
        connections dictionary.
        In any case (i.e. whether the interaction was positive or not), the 
        payoff is added to the reputations entry of the opponent.
        """
        self.wealth.append(float(payoff))
        self.payoff_history.append((float(payoff)))
        id_opponent = opponent.ident
        region_opponent = opponent.place
        if id_opponent in self.reputations:
            pass
        else:
            self.reputations[id_opponent] = []
        self.reputations[id_opponent].append(float(payoff))

        if payoff > parameter_file['payoffs']['c']:
            """In this case the interaction is considered positive."""
            if region_opponent not in self.connections.keys():
                self.connections[region_opponent] = set()
            self.connections[region_opponent].add(id_opponent)

        else:
            """In this case the interaction is considered negative."""
            if region_opponent in self.connections.keys():
                """Remove opponent from connections and, if she was the last 
                connection in region, the whole region."""
                if id_opponent in self.connections[region_opponent]:
                    self.connections[region_opponent].remove(id_opponent)
                    if not self.connections[region_opponent]:
                        del self.connections[region_opponent]
                        assert region_opponent not in self.connections.keys(), \
                            "Error in removing region!"

            """Add agent to the personal gossip list."""
            if id_opponent not in self.gossip:
                self.gossip[id_opponent] = []
            self.gossip[id_opponent].append(self.control_1)

            """Add agent to the gossip list of the partners."""
            for a in self.reputations.keys(): 
                # For all agents with whom I have a positive last interactions...
                if self.reputations[a][-1] > parameter_file['payoffs']['c']:
                    self.model.get_agent(a).receive_gossip(id_opponent)
    
    def receive_gossip(self, id_oppo):
        """Receive information about defecting agents
        
        The agent receives gossip by another agent by writing the gossip 
        rejection probability (parameter `control_2`) to her gossip dictionary.
        
        Parameters
        ----------
        id_oppo: int
            Identification number of the agent she hears gossip about

        Returns
        -------
        None        
        """
        assert isinstance(id_oppo, int), \
            "Gossip dict must only contain ids of agents"
        if id_oppo not in self.gossip.keys():
            self.gossip[id_oppo] = []
        self.gossip[id_oppo].append(self.control_2)

    def change_strat(self, newstrat, parameterfile, 
                     trust_shock=False, control_shock=False):
        """Updates the strategy of an agent

        Updates the strategy of the agent. Called during selection phase.
        The values of trust and control are set in accordance to the initial 
        values set for those parameters in the parameter file, or, if there is
        a shock, to the shock values (usually zero).
        For defectors, trust is always `np.nan`. 
        Control is drawn from a binomial distribution according to the 
        parameter values for `C/D_share_full_control_1/2`.
        # TODO Explain why
        
        Parameters
        ----------
        newstrat : [type]
            [description]
        parameterfile : [type]
            [description]
        trust_shock : bool, optional
            If True, trust for cooperators is set to zero, by default False
        control_shock : bool, optional
            If True, control for cooperators is set to zero, by default False
        """
        oldstrat = self.strategy
        self.strategy = newstrat
        del oldstrat
        self.has_changed_strat = True
        if newstrat == "ALL_C":
            if control_shock:
                self.control_1 = parameterfile["c_shock_value"]
                self.control_2 = parameterfile["c_shock_value"]
            else:
                self.control_1 = parameterfile["C_control_1"]
                self.control_2 = parameterfile["C_control_2"]
            if trust_shock:
                self.trust = parameterfile["t_shock_value"]
            else:
                self.trust = parameterfile["C_trust"]
            self.interval = np.NaN
        else:
            self.trust = np.nan
            self.control_1 = parameterfile["D_control_1"]
            self.control_2 = parameterfile["D_control_1"]
            self.interval = parameterfile["D_interval"]
        
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
    