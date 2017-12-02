import numpy
import math
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

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
        self.model.add(Dense(10, input_dim=5, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.weightInit = self.model.get_weights()
        self.targets=numpy.zeros(1).reshape(1,1)
        self.states=numpy.zeros((1,5)).reshape(1,5)
        self.hasRun = False

    def choose_action(self, state, trial):
        #self.explore=max(self.MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((trial+1)/25)))
        ep = max((trial-1)//100,0)
        self.explore = max(1/pow(2,ep),self.MIN_EXPLORE_RATE)
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


    def save_q(self, old_state, new_state, reward, action, trial):
        #self.learn=max(self.MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((trial+1)/25)))
        ep = max((trial-1)//100,0)
        self.learn = max(1/pow(2,ep),self.MIN_LEARNING_RATE)
        q0 = self.model.predict(numpy.append(new_state,[0]).reshape(1,5))
        q1 = self.model.predict(numpy.append(new_state,[1]).reshape(1,5))
        oldQ = self.model.predict(numpy.append(old_state,[action]).reshape(1,5))
        qTarget = (1 - self.learn)*oldQ + self.learn*(reward + self.DISCOUNT_FACTOR*max(q0,q1))
        
        # Use all data to train NN
        if trial == 0 and self.hasRun == False:
            self.targets=numpy.array([qTarget]).reshape(1,1)
            self.states=numpy.append(old_state,[action]).reshape(1,5)
            self.hasRun = True
        else:
            self.targets=numpy.append(self.targets, numpy.array([qTarget]).reshape(1,1), axis=0)
            self.states=numpy.append(self.states, numpy.append(old_state,[action]).reshape(1,5), axis=0)
            
    def train_q(self):
        self.model.set_weights(self.weightInit)
        self.model.fit(self.states,self.targets,epochs=10,batch_size=1, verbose=0)
