import numpy
import math
import random
from keras.models import Sequential
from keras.layers import Dense

class QAgent(object):
    """Generic Q-Learning Agent
    Description of class goes here.
    """

    def __init__(self):
        self.ACTION_SPACE = 2
        self.DISCOUNT_FACTOR = 0.99
        self.learn = 1
        self.explore = 1

        self.MIN_EXPLORE_RATE = 0.01
        self.MIN_LEARNING_RATE = 0.1
        self.model = Sequential()
        self.model.add(Dense(12, input_dim=5, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.targets=numpy.zeros(1).reshape(1,1)
        self.states=numpy.zeros((1,5)).reshape(1,5)
        self.ttest=0

    def choose_action(self, state, trial):
        self.explore=max(self.MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((trial+1)/25)))
        if random.random() < self.explore:
            action = random.randint(0,1)
        else:
            q0 = self.model.predict(numpy.append(state,[0]).reshape(1,5))
            q1 = self.model.predict(numpy.append(state,[1]).reshape(1,5))
            if q0 > q1:
                action = 0
            else:
                action = 1
        return action


    def update_q(self, old_state, new_state, reward, action, trial):
        self.learn=max(self.MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((trial+1)/25)))
        q0 = self.model.predict(numpy.append(new_state,[0]).reshape(1,5))
        q1 = self.model.predict(numpy.append(new_state,[1]).reshape(1,5))
        qTarget = self.learn*(reward + self.DISCOUNT_FACTOR*max(q0,q1))
        #self.model.fit(numpy.append(old_state,[action]).reshape(1,5),numpy.array([qTarget]).reshape(1,1), epochs=50, batch_size=1, verbose=0)
        self.targets=numpy.append(self.targets, numpy.array([qTarget]).reshape(1,1), axis=0)
        self.states=numpy.append(self.states, numpy.append(old_state,[action]).reshape(1,5), axis=0)
        if self.ttest != trial:
            self.model.fit(self.states,self.targets,epochs=1,batch_size=1, verbose=0)
            self.ttest = trial
