import gym
import numpy as np
from dql import DQN



# parameters needed for the training of the entire game
opts = dict(
    num_episodes=10,   # number of games the agent will play
    num_iter=1000,
    is_training=False,
    input_img_dim=(96, 96, 3),
    temporal_window=1,
    render=True,
    learn_freq=5,
    save_model_freq=1200,
    net_opts=[]    # information about the DQN network
)

env = gym.make("CarRacing-v0")   # what ever the game is the idea is to make the code easily applicable to other games
observation = env.reset()
episode_count=0

dqn = DQN(opts)
print(opts)
print(observation.shape)
action = dqn.policy([])
observation, reward, done, info = env.step(action)
print(observation.shape)
print(reward)
print(done)
print(info)
while episode_count < opts['num_episodes'] and opts['is_training']:

    for _ in range(opts['num_iter']):

        if(opts['render']):
            env.render()
        
        # TODO add the learning module
        #action = env.action_space.sample()   # currently our agent gets random action 
        #
        action = dqn.policy([])
        
        observation, reward, done, info = env.step(action)
        # print(observation, reward)

    if episode_count % opts['save_model_freq']:
        print('checkpoint:>> save current model')

    episode_count += 1


print('<<<<<<<<<<< Agent action Terminated >>>>>>>>>>>>>>>>>')