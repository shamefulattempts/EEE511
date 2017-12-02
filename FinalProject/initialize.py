import gym
from QLearnAgent import QAgent
from time import sleep

env = gym.make('CartPole-v1')
print(list(zip(env.observation_space.low, env.observation_space.high)))
agent = QAgent()
highscore = 0
best_trial = 0
for trial in range(2000): # run 20 episodes
    observation = env.reset()
    time=0
    while True: # run until episode is done
        #sleep(1)
        #env.render()
        action = agent.choose_action(observation,trial)
        old_obs = observation
        observation, reward, done, info = env.step(action)
        # Update Q values at each time step
        agent.update_q(old_obs,observation,reward,action,trial)
        time += 1
        #print(observation)
        if done:
            print("Trial %d lasted %d time steps Learning Rate: %f Explore Rate: %f" % (trial, time, agent.learn, agent.explore))
            # Update Q values after every trial
            #agent.update_q(old_obs,observation,reward,action,trial)
            if (time > highscore):
            	highscore = time
            	best_trial = trial
            break

print("Best time %d time steps at trial %d" % (highscore, best_trial))
