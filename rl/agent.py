from tensorflow.keras.layers import Activation, Dense, GRU
from tensorflow.keras.models import Sequential
from collections import deque

import numpy as np
import random

class Agent:
    def __init__(self, state_size, amount_features, learning=True, epsilon=1.0, epsilon_min=0.01, epsilon_decay=-0.015, gamma=0.95):
        self.__state_size = state_size
        self.__amount_features = amount_features
        self.__action_size = 3
        self.memory = deque(maxlen=1000)
        self.__gamma = gamma
        self.__epsilon = epsilon
        self.__epsilon_min = epsilon_min
        self.__epsilon_decay = epsilon_decay
        self.__t = 0
        self.__learning = learning
        self.inventory = []
        self.__model = self.__model()
    
    def __model(self):
        model = Sequential()

        model.add(GRU(
            units=64,
            activation="tanh",
            return_sequences=True,
            dropout=0.2,
            input_shape=(self.__state_size, self.__amount_features)
            )
        )

        model.add(GRU(
            units=64,
            activation="tanh",
            return_sequences=False,
            dropout=0.2
            )
        )

        model.add(Dense(32, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(self.__action_size, activation="linear"))

        model.compile(
            optimizer="Nadam",
            loss="mse"
        )

        return model

    def act(self, state):
        if self.__learning and random.random() <= self.__epsilon:
            return random.randrange(self.__action_size)

        options = self.__model.predict(state)
        print("Options: ", options)
        # Return the index of the max value
        return np.argmax(options[0])

    def replay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.__gamma * np.amax(self.__model.predict(next_state)[0])
            
            target_f = self.__model.predict(state)
            target_f[0][action] = target
            self.__model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.__epsilon > self.__epsilon_min:
            self.__epsilon = math.exp(self.__epsilon_decay * self.__t)
            self.__t =  self.__t + 1
