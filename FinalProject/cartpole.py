import gym
from QLearnAgent import QAgent

env = gym.make('CartPole-v0')
agent = QAgent()
highscore = 0
for i_episode in range(2000): # run 20 episodes
    observation = env.reset()
    points = 0 # keep track of the reward each episode
    while True: # run until episode is done
        env.render()
        action = agent.choose_action(observation)
        old_obs = observation
        observation, reward, done, info = env.step(action)
        agent.update_q(old_obs,observation,reward,action)
        points += reward
        #print(observation)
        if done:
            print("Episode %d finished with score of %f" % (i_episode, points))
            if points > highscore: # record high score
                highscore = points
            break
