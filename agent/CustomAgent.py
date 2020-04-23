try:
    from runner.abstracts import Agent
except:
    class Agent(object): pass
import random
import numpy as np
import torch

class CustomAgent(Agent):
    '''
    An example agent that just output a random action.
    '''
    def __init__(self, *args, **kwargs):
        '''
        [OPTIONAL]
        Initialize the agent with the `test_case_id` (string), which might be important
        if your agent is test case dependent.
        
        For example, you might want to load the appropriate neural networks weight 
        in this method.
        '''
        test_case_id = kwargs.get('test_case_id')
        '''
        # Uncomment to help debugging
        print('>>> __INIT__ >>>')
        print('test_case_id:', test_case_id)
        '''
        # initiate the preference matrix
        self.pref = self.init_pref(10,50)

    # up to you to change, even use a NN for this if you want to
    def init_pref(self,x,y):
        temp = np.zeros((x,y))

        def l(num):
            return max(0,num)

        def u(num):
            return min(num,x-1)
        temp[0][0] = 1
        for j in range(1,y):
            for i in range(0,min(j+1,x)):
                temp[i][j] = 0.2*(temp[l(i-1)][l(j-1)]+temp[u(i+1)][l(j-1)]+temp[i][l(j-1)]+temp[i][l(j-2)]+temp[i][l(j-3)])

        return temp


    def initialize(self, **kwargs):
        '''
        [OPTIONAL]
        Initialize the agent.

        Input:
        * `fast_downward_path` (string): the path to the fast downward solver
        * `agent_speed_range` (tuple(float, float)): the range of speed of the agent
        * `gamma` (float): discount factor used for the task

        Output:
        * None

        This function will be called once before the evaluation.
        '''
        fast_downward_path  = kwargs.get('fast_downward_path')
        agent_speed_range   = kwargs.get('agent_speed_range')
        gamma               = kwargs.get('gamma')
        '''
        # Uncomment to help debugging
        print('>>> INITIALIZE >>>')
        print('fast_downward_path:', fast_downward_path)
        print('agent_speed_range:', agent_speed_range)
        print('gamma:', gamma)
        '''
    
    def get_agent_pos(self,state):
        for i in range(10):
            for j in range(50):
                if state[1][i][j] > 1:
                    return i,j

    # compute new location after action
    def new_pos(self,x,y,a):
        
        def l(num):
            return max(0,num)

        def u(num):
            return min(num,9)

        if a == 0:
            return l(x-1),l(y-1)
        elif a == 1:
            return u(x+1),l(y-1)
        elif a == 2:
            return x,l(y-1)
        elif a == 3:
            return x,l(y-2)
        else:
            return x,l(y-3)


    def compute_p_helper(self,original,new,car,maxCar,speedRange,agent=0):
        if car == maxCar:
            p_action = 1/(speedRange[1]-speedRange[0]+1)
            p_product = 1
            for car in range(maxCar):
                p_product *= p_action
                if car != 0:
                    if new[car] <= new[car-1]:
                        return 0
                    elif new[car] == (new[car-1]+1):
                        p_product *= (new[car]-(original[car]+speedRange[0])+1)
            return p_product
        else:
            sum_ = 0
            minSpeed = speedRange[1]
            maxSpeed = speedRange[0]
            for speed in range(maxSpeed,minSpeed+1):
                new[car] = original[car]+speed
                if agent >= new[car] and agent <= (original[car]+minSpeed):
                    continue
                sum_ += self.compute_p_helper(original,new,car+1,maxCar,speedRange,agent)
            return sum_

    def compute_p(self,x,y,state):
        lane = state[0][x]
        cut = lane[y:y+4]
        if (y+4)>len(lane):
            cut = np.append(cut,lane[:(y+4)%len(lane)])
        original = []    
        for i in range(len(cut)):
            if cut[i] == 1:
                original.append(i)
        print(cut,original)
        original = np.array(original,dtype=np.intc)
        new = np.zeros(len(original),dtype=np.intc)
        # get speed range from the environment
        speedRange = [-3,-1]
        return self.compute_p_helper(original,new,0,len(original),speedRange,0)

    def step(self, state, epsilon):
        ''' 
        [REQUIRED]
        Step function of the agent which computes the mapping from state to action.
        As its name suggests, it will be called at every step.
        
        Input:
        * `state`:  tensor of dimension `[channel, height, width]`, with 
                    `channel=[cars, agent, finish_position, occupancy_trails]`

        Output:
        * `action`: `int` representing the index of an action or instance of class `Action`.
                    In this example, we only return a random action
        '''
        '''
        # Uncomment to help debugging
        print('>>> STEP >>>')
        print('state:', state)
        '''

        Q_values = self.model.forward(state)
        x,y = get_agent_pos(state)
        p_noCollision = np.zeros(5)
        pref = np.zeros(5)
        for action in range(5):
            x1,y1 = self.new_pos(x,y,action)
            p_noCollision[action] = self.compute_p(x1,y1,state)
            pref[action] = self.pref[x1][y1]

        final_term = Q_values + epsilon*torch.tensor(p_noCollision) + min(1-epsilon,0.3)*torch.tensor(pref)
        
        idx = torch.argmax(final_term).item()

        sample = random.random()
        if sample < epsilon:
            idx = random.randrange(5)
        return idx  

    def update(self, *args, **kwargs):
        '''
        [OPTIONAL]
        Update function of the agent. This will be called every step after `env.step` is called.
        
        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with 
                   `channel=[cars, agent, finish_position, occupancy_trails]`
        * `action` (`int` or `Action`): the executed action (given by the agent through `step` function)
        * `reward` (float): the reward for the `state`
        * `next_state` (same type as `state`): the next state after applying `action` to the `state`
        * `done` (`int`): whether the `action` induce terminal state `next_state`
        * `info` (dict): additional information (can mostly be disregarded)

        Output:
        * None

        This function might be useful if you want to have policy that is dependant to its past.
        '''
        state       = kwargs.get('state')
        action      = kwargs.get('action')
        reward      = kwargs.get('reward')
        next_state  = kwargs.get('next_state')
        done        = kwargs.get('done')
        info        = kwargs.get('info')
        '''
        # Uncomment to help debugging
        print('>>> UPDATE >>>')
        print('state:', state)
        print('action:', action)
        print('reward:', reward)
        print('next_state:', next_state)
        print('done:', done)
        print('info:', info)
        '''
