import gym
from QLearnAgent import QAgent
from time import sleep
import random
env = gym.make('CartPole-v1')
#agent = QAgent(seed)
#highscore = 0
#best_trial = 0
num_runs = 5
sensor_noise=[0, 0.05*random.uniform(-0.5,0.5), 0.1*random.uniform(-0.5,0.5), random.gauss(0,.1), random.gauss(0,.2)]
text_file=open("output.txt","w+")
for i in range(len(sensor_noise)):
    if i == 0:
        print("No noise")
        text_file.write("No noise")
    elif i == 1:
        print("5 pct uniform sensor noise")
        text_file.write("5 pct uniform sensor noise")
    elif i == 2:
        print("10 pct uniform sensor noise")
        text_file.write("10 pct uniform sensor noise")
    elif i == 3:
        print("0.1 var gaussian sensor noise")
        text_file.write("0.1 var gaussian sensor noise")
    elif i == 4:
        print("0.2 var gaussian sensor noise")
        text_file.write("0.2 var gaussian sensor noise")
    else:
        print("ERROR")
    wins = 0
    pct = 0
    summation = 0
    for run in range (num_runs):
        agent = QAgent()
        highscore = 0
        best_trial = 0
        for trial in range(2000): # run 20 episodes
            observation = env.reset()
            time=0
            while True: # run until episode is done
                #if trial%100 == 0:
                #env.render()
                #sleep(1)
                #env.render()
                action = agent.choose_action(observation,trial)
                observation[2]=observation[2]*(1+sensor_noise[i]) #Adding sensor noise
                old_obs = observation
                observation, reward, done, info = env.step(action)
                agent.update_q(old_obs,observation,reward,action,trial)
                time += 1
                #print(observation)
                if done:
                    #print("Trial %d lasted %d time steps Learning Rate: %f Explore Rate: %f" % (trial, time, agent.learn, agent.explore))
                    #if trial%500 ==0:
                        #print("Trial %d lasted %d time steps Learning Rate: %f Explore Rate: %f" % (trial, time, agent.learn, agent.explore))
                    if (time > highscore):
                        highscore = time
                        best_trial = trial
                        #print("HIGHSCORE: Trial %d lasted %d time steps Learning Rate: %f Explore Rate: %f" % (trial, time, agent.learn, agent.explore))
                    break
            if highscore >= 5999:
                break
        print("Best time %d time steps at trial %d" % (highscore, best_trial))
        text_file.write("Best time %d time steps at trial %d" % (highscore, best_trial))
        if highscore >= 5999:
            summation += best_trial
            wins += 1
    avg=summation/wins
    pct=wins/num_runs
    print("Average success in %f trials with a win percentage of %f" % (avg, pct))
    text_file.write("Average success in %f trials with a win percentage of %f" % (avg, pct))
