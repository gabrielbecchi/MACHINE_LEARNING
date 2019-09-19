#Alpha e Gamma adaptativos
#Pretreining: com desvio padrao
#Google Colab
#MMenge - Ranking por experimento fatorial
#Pendulo com dois elos

import numpy as np
import matplotlib.pyplot as plt
import gym
import time
import pandas as p
from dql import deep_q_learning

class CartPole:
	def __init__(self):
		self.env = gym.make('CartPole-v0')
		self.machine = deep_q_learning(self.env)	

	def run(self,episodes):
		#plt.axis([0,1000, 0, 350])
		dic_results = {0:0,100:0,200:0}
		continuous_200 = 0
		aux = 0
		for ep in range(episodes):
			aux += 1
			#print("EPISODE: "+str(ep+1))
			observation = self.env.reset().round(3)[2:]
			#input(observation)
			done = False
			total_reward = 0
			max_iterations = 0
			#Reset Train DF
			self.machine.new_episode()
			while(not(done)):
				#Choose Action
				#observation = observation.round(3)
				action = self.machine.choose_action(observation)
				#Take Action
				observation, reward, done, info = self.env.step(action)

				#Set Reward
				observation = observation.round(3)[2:]
				self.machine.set_reward(reward,observation,done)
				
				max_iterations += 1
				#if(aux%10 == 0):
				#self.env.render()

			#Reshape Q_Value
			self.machine.end_episode()
			
			if(ep % 1000 == 0):
				print(dic_results)
				dic_results = {0:0,100:0,200:0}
			if(max_iterations < 100):
				dic_results[0] += 1
				continuous_200 = 0
				continue
			if(max_iterations < 199):
				dic_results[100] += 1
				continuous_200 = 0
				continue
			#Sucess
			dic_results[200] += 1
			continuous_200 += 1
			if(continuous_200 >= 100):
				print("Finished EP: "+str(ep))
				break
			continue

	def save_machine(self):
		self.machine.save_machine()

machine = CartPole()
machine.run(5000)
#machine.save_machine()

