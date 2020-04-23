try:
    from runner.abstracts import Agent
except:
    class Agent(object): pass
import random
import math
import numpy as np

class ReplayBuffer():
    def __init__(self, buffer_limit=buffer_limit):
        self.buffer_limit = buffer_limit
        self.buffer = []
        self.position = 0
    
    def push(self, transition):
        if len(self.buffer) < self.buffer_limit:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position+1)%self.buffer_limit
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer,batch_size)
        t_batch = list(map(lambda x: torch.tensor(list(x),dtype=torch.float,device=device),zip(*batch)))
        t_batch[1] = t_batch[1].type(torch.int)
        return tuple(t_batch)

    def __len__(self):
        return len(self.buffer)


class ExampleAgent(Agent):
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

    def step(self, state, *args, **kwargs):
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
        epsilon = 1.00

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


def create_agent(test_case_id, *args, **kwargs):
    '''
    Method that will be called to create your agent during testing.
    You can, for example, initialize different class of agent depending on test case.
    '''
    return ExampleAgent(test_case_id=test_case_id)


if __name__ == '__main__':
    import sys
    import time
    from env import construct_task2_env

    FAST_DOWNWARD_PATH = "/fast_downward/"

    def test(agent, env, runs=1000, t_max=100):
        rewards = []
        for run in range(runs):
            state = env.reset()
            agent_init = {'fast_downward_path': FAST_DOWNWARD_PATH, 'agent_speed_range': (-3,-1), 'gamma' : 1}
            agent.initialize(**agent_init)
            episode_rewards = 0.0
            for t in range(t_max):
                action = agent.step(state)   
                next_state, reward, done, info = env.step(action)
                full_state = {
                    'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 
                    'done': done, 'info': info
                }
                agent.update(**full_state)
                state = next_state
                episode_rewards += reward
                if done:
                    break
            rewards.append(episode_rewards)
        avg_rewards = sum(rewards)/len(rewards)
        print("{} run(s) avg rewards : {:.1f}".format(runs, avg_rewards))
        return avg_rewards

    def timed_test(task):
        start_time = time.time()
        rewards = []
        for tc in task['testcases']:
            agent = create_agent(tc['id'])
            print("[{}]".format(tc['id']), end=' ')
            avg_rewards = test(agent, tc['env'], tc['runs'], tc['t_max'])
            rewards.append(avg_rewards)
        point = sum(rewards)/len(rewards)
        elapsed_time = time.time() - start_time

        print('Point:', point)

        for t, remarks in [(0.4, 'fast'), (0.6, 'safe'), (0.8, 'dangerous'), (1.0, 'time limit exceeded')]:
            if elapsed_time < task['time_limit'] * t:
                print("Local runtime: {} seconds --- {}".format(elapsed_time, remarks))
                print("WARNING: do note that this might not reflect the runtime on the server.")
                break

    def get_task():
        tcs = [('task_2_tmax50', 50), ('task_2_tmax40', 40)]
        return {
            'time_limit': 600,
            'testcases': [{ 'id': tc, 'env': construct_task2_env(), 'runs': 300, 't_max': t_max } for tc, t_max in tcs]
        }

    #task = get_task()
    #timed_test(task)

    