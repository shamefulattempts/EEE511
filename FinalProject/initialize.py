import gym
from QLearnAgent import QAgent
from time import sleep

env = gym.make('CartPole-v1')
#agent = QAgent(seed)
#highscore = 0
#best_trial = 0
summation = 0
for run in range (10):
    agent = QAgent()
    highscore = 0
    best_trial = 0
    for trial in range(1000): # run 20 episodes
        observation = env.reset()
        time=0
        while True: # run until episode is done
            #if trial%100 == 0:
                #env.render()
            #sleep(1)
            #env.render()
            action = agent.choose_action(observation,trial)
            old_obs = observation
            observation, reward, done, info = env.step(action)
            agent.update_q(old_obs,observation,reward,action,trial)
            time += 1
            #print(observation)
            if done:
                #print("Trial %d lasted %d time steps Learning Rate: %f Explore Rate: %f" % (trial, time, agent.learn, agent.explore))
                #if trial%100 ==0:
                    #print("Trial %d lasted %d time steps Learning Rate: %f Explore Rate: %f" % (trial, time, agent.learn, agent.explore))
                if (time > highscore):
                    highscore = time
                    best_trial = trial
                    #print("HIGHSCORE: Trial %d lasted %d time steps Learning Rate: %f Explore Rate: %f" % (trial, time, agent.learn, agent.explore))
                break

    print("Best time %d time steps at trial %d" % (highscore, best_trial))
    summation += highscore
avg=summation/10
print(avg)
