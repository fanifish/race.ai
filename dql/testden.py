import numpy as np
from dql import DQN

def test_dqn_init():
    dqn = DQN((96, 96, 3))
    print('test done')


if __name__ == '__main__':
    test_dqn_init()