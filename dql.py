#
#
#@author Faniel Ghirmay
# a python implementation of DQn learnig architecture
#
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


class DQN:
    def __init__(self, opts):
        print('init Dqn .. ')
        self.input_dim = opts['input_img_dim']

        # might have to insist this network only works with volumes
        self.state_size = self.input_dim[0]*self.input_dim[1]*self.input_dim[2]
        self.temporal_window = opts['temporal_window']
        self.num_actions = 3 #TODO get num of actions from options
        self.num_input = (self.state_size*self.temporal_window) + (self.num_actions*self.temporal_window) + self.state_size 
        self.learn_freq = opts['learn_freq']
        self.network = self.create_network(opts['net_opts'])
        print('Done init')

    def create_network(self, net_opts):
        model = Sequential()
        model.add(Dense(self.num_input))
        model.add(Dense(2000, activation="relu"))
        model.add(Dense(200, activation="relu"))
        model.add(Dense(3, activation="softmax"))
        return model

    # TODO add epsilon  tradeoff btn exploitation and exploration
    def policy(self, s):
        action = self.network.predict(s)
        print(action)
        return action

    def forward(self, input_img):
        print('training')
