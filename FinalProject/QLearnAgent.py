import numpy
import math
import random

class QAgent(object):
    """Generic Q-Learning Agent
    Description of class goes here.
    """

    def __init__(self):
        self.STATE_BINS = (4, 3, 10, 4) # (x, x', theta, theta')
        self.ACTION_SPACE = 2
        self.STATE_BOUNDS = numpy.zeros((4,2))
        self.STATE_BOUNDS[0] = (-2.4, 2.4)
        self.STATE_BOUNDS[1] = (-2, 2)
        self.STATE_BOUNDS[2] = (-math.radians(15), math.radians(15))
        self.STATE_BOUNDS[3] = (-3, 3)
        self.DISCOUNT_FACTOR = 0.99
        self.LEARNING_RATE = 0.5
        self.EXPLORE_RATE = 0.1
        
        self.q_table=numpy.zeros(self.STATE_BINS + (self.ACTION_SPACE,))

    def choose_action(self, state):
        dis_state = self.__discretize_state(state)
        if random.random() < self.EXPLORE_RATE:
            action = random.randint(0,1)
        else:
            action = numpy.argmax(self.q_table[dis_state])
        return action

    def __discretize_state(self,state):
        bin_indice = []
        bin_index = 0
        for i in range(len(state)):
            if state[i] <= self.STATE_BOUNDS[i][0]:
                bin_index = 0
            elif state[i] >= self.STATE_BOUNDS[i][1]:
                bin_index = self.STATE_BINS[i] - 1
            else:
                # Mapping the state bounds to the bucket array
                bound_width = self.STATE_BOUNDS[i][1] - self.STATE_BOUNDS[i][0]
                offset = (self.STATE_BINS[i]-1)*self.STATE_BOUNDS[i][0]/bound_width
                scaling = (self.STATE_BINS[i]-1)/bound_width
                bucket_index = int(round(scaling*state[i] - offset))
            bin_indice.append(bin_index)
        return tuple(bin_indice)

    def update_q(self, old_state, new_state, reward, action):
        d_old_state = self.__discretize_state(old_state)
        d_new_state = self.__discretize_state(new_state)
        max_q = numpy.amax(self.q_table[d_new_state])
        self.q_table[d_old_state +(action,)] += self.LEARNING_RATE*(reward + self.DISCOUNT_FACTOR*max_q - self.q_table[d_old_state + (action,)])

