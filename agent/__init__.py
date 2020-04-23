from CustomAgent import CustomAgent
import collections
import random
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

#Hyperparameters
learning_rate  = 0.001
batch_size     = 32
max_episodes   = 10000
t_max          = 600
min_epsilon    = 0.01
epsilon_decay  = 500
print_interval = 50
target_update  = 20
train_steps    = 10
buffer_limit   = 5000
min_buffer     = 2000
gamma          = 0.98

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


def create_agent(test_case_id, *args, **kwargs):
    '''
    Method that will be called to create your agent during testing.
    You can, for example, initialize different class of agent depending on test case.
    '''
    return CustomAgent(test_case_id=test_case_id, *args, **kwargs)


if __name__ == '__main__':
    import sys
    import time
    from env import construct_task2_env

    FAST_DOWNWARD_PATH = "/fast_downward/"

    def compute_loss(model, target, states, actions, rewards, next_states, dones):
        HuberLoss = nn.SmoothL1Loss()
        estimations = model(states).gather(1,actions.type(torch.long))
        with torch.no_grad():
            targets = target(next_states).detach().max(1)[0].unsqueeze(1)
        targets = rewards + (gamma * targets * (1-dones))
        return HuberLoss(estimations,targets)

    def optimize(model, target, memory, optimizer):
        batch = memory.sample(batch_size)
        loss = compute_loss(model, target, *batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def compute_epsilon(episode):
        epsilon = math.exp(-1. * episode / epsilon_decay)
        if epsilon < min_epsilon:
            return min_epsilon
        else:
            return epsilon

    def train(model_class, env):
        # Initialize model and target network
        model = model_class(env.observation_space.shape, env.action_space.n).to(device)
        target = model_class(env.observation_space.shape, env.action_space.n).to(device)
        target.load_state_dict(model.state_dict())
        target.eval()

        # Initialize replay buffer
        memory = ReplayBuffer()

        print(model)

        # Initialize rewards, losses, and optimizer
        rewards = []
        losses = []
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for episode in range(max_episodes):
            epsilon = compute_epsilon(episode)
            state = env.reset()
            episode_rewards = 0.0

            for t in range(t_max):
                # Model takes action
                action = model.step(state, epsilon)

                # Apply the action to the environment
                next_state, reward, done, info = env.step(action)

                # Save transition to replay buffer
                memory.push(Transition(state, [action], [reward], next_state, [done]))

                state = next_state
                episode_rewards += reward
                if done:
                    break
            rewards.append(episode_rewards)
            
            # Train the model if memory is sufficient
            if len(memory) > min_buffer:
                if np.mean(rewards[print_interval:]) < 0.1:
                    print('Bad initialization. Please restart the training.')
                    exit()
                for i in range(train_steps):
                    loss = optimize(model, target, memory, optimizer)
                    losses.append(loss.item())

            # Update target network every once in a while
            if episode % target_update == 0:
                target.load_state_dict(model.state_dict())

            if episode % print_interval == 0 and episode > 0:
                print("[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                                episode, np.mean(rewards[print_interval:]), np.mean(losses[print_interval*10:]), len(memory), epsilon*100))
        return model

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

    