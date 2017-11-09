import gym
import time


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())
        time.sleep(.05)

#import gym env = gym.make('CartPole-v0') env.reset() env.render()
