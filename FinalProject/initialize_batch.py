import gym
from QLearnAgent_batch import QAgent
from time import sleep

env = gym.make('CartPole-v1')
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
        agent.save_q(old_obs,observation,reward,action,trial)
        time += 1
        #print(observation)
        if done:
            print("Trial %d lasted %d time steps Learn Rate: %f Explore Rate: %f" % (trial, time, agent.learn, agent.explore))
            # Update Q values after every trial
            if (trial%100 == 0 and trial != 0):
                print("Training...")
                agent.train_q()
            if (time > highscore):
            	highscore = time
            	best_trial = trial
            break

print("Best time %d time steps at trial %d" % (highscore, best_trial))
