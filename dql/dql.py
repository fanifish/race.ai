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
     #   self.input_dim = opts['input_img_dim']
     #   self.temporal_window = opts['temporal_window']
     #   self.learn_freq = opts['learn_freq']
        self.num_seq = 0
        self.network = self.create_network(opts['net_opts'])
        print('Done init')

    # takes in network architecture info as input
    def create_network(self, net_opts):
        model = Sequential()
        # TODO for futer use net_opts to create the narchitecture

        model.add(Dense(2000, input_shape=(96, 96, 3))) # pass input shape size will depend on the temporal window

        model.add(Dense(200))


        model.add(Dense(3)) # [steer, gas, brake]

        return model

    def policy(self, s):
        return [0, 1, 0]


    def forward(self, s):
        self.num_seq += 1
