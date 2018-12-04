#
#
#@author Faniel Ghirmay
# a python implementation of DQn learnig architecture
#
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


class DQN:
    def __init__(self, input_dim):
        print('init Dqn .. ')
        self.num_input = 90
        self.input_dim = input_dim
        print('Done init')

    def create_network(self):
        print('network created')
        return None
