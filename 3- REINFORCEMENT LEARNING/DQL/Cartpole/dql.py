from dnn import dnn
import pandas as p
import random
from ast import literal_eval
import numpy as np
import os
import sys

#HYPER PARAMETERS
#DNN_Number_Layers = 
#DNN_Units_Layer =
#EXPLORATION_Decay = 
#PRETRAIN_Observation = 
#ALPHA =
#GAMMA = 
#LR = 

#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#from matplotlib import style

#LOAD_MODEL = True
LOAD_MODEL = False

class deep_q_learning():
	def __init__(self,env):
		self.env = env
		self.model = dnn()
		if(LOAD_MODEL == False):
			#input_dim,output_dim,number_layers,units_per_layer
			self.model.create_model(2,2,3,4)
		else:
			self.model.load_model()

		self.exploration_rate = 1
		self.exploration_decrease = 0.00005
		self.exploration_minimal = 0.025
		self.alpha = 0.10
		self.gamma = 0.99
		self.pretrain_observations = 5000
		self.number_episodes = 0
	
		self.train_df = p.DataFrame(columns=["Observation","Q_Values","Action","Reward","new_Observation"])

		self.pretrain = False
		#if(LOAD_MODEL == False):
		#	self.pre_train()

	def pre_train(self):
		if(os.path.isfile("pretrain_data.csv") == False):
			self.pretrain = True
			print("Pretrain Round")
			self.pretrain_df = p.DataFrame(columns=["Observation","Q_Values"])
		else:
			self.pretrain = False
			self.pretrain_df = p.read_csv("pretrain_data.csv",converters={"Observation":literal_eval,"Q_Values":literal_eval})
			x = p.DataFrame(self.pretrain_df.Observation.values.tolist())
			y = p.DataFrame(self.pretrain_df.Q_Values.values.tolist())
			self.model.pretrain_model(x,y)


	def new_episode(self):
		self.accumulative_reward = 0
		self.episode_interations = 0
		self.number_episodes += 1
		return


	def choose_action(self,observation):
		self.data_serie = p.Series()
		self.data_serie["Observation"] = list(observation)
		if(random.random() < self.exploration_rate):
			self.data_serie["Q_Values"] = [0.1*random.uniform(-1, 1),0.1*random.uniform(-1, 1)]
			action = self.env.action_space.sample()
		else:
			Q_Values = list(self.model.predict(observation))
			self.data_serie["Q_Values"] = Q_Values
			action = Q_Values.index(max(Q_Values))
		self.data_serie["Action"] = action
		return action

	def set_reward(self,reward,new_obervation,done):
		self.episode_interations += 1
		self.data_serie["new_Observation"] = new_obervation
		#Episode not ended
		if(not(done)):
			new_Q_Values = list(self.model.predict(new_obervation))
			self.data_serie["Reward"] = (1+self.gamma*round(max(new_Q_Values),8))/10
		#Episode End
		else:
			if(self.episode_interations < 195):
				self.data_serie["Reward"] = 0
			else:
				self.data_serie["Reward"] = 1
			self.data_serie["new_Observation"] = None
		
		self.train_df = self.train_df.append(self.data_serie,ignore_index=True)	
		if(self.number_episodes%10 == 0):
			self.print_instance()

		if(self.pretrain):
			self.data_serie["Q_Values"] = [0.1*random.uniform(-1, 1),0.1*random.uniform(-1, 1)]
			self.pretrain_df = self.pretrain_df.append(self.data_serie[["Observation","Q_Values"]],ignore_index=True)
			print(len(self.pretrain_df))
			if(len(self.pretrain_df) >= self.pretrain_observations):
				print("End of pretrain")
				self.pretrain_df.to_csv("pretrain_data.csv",index=False)
				sys.exit()

	def print_instance(self):
		#print(str(self.data_serie["Observation"])+" - "+str(self.data_serie["new_Observation"]))
		print(str(round(max(self.data_serie["Q_Values"]),8))+" -> ",end='')
		print(str(round(self.data_serie["Reward"],5))+" = "+str(self.data_serie["Action"]))

	def end_episode(self):
		#Update Explore/Exploid
		if(self.exploration_rate > self.exploration_minimal):
			self.exploration_rate = self.exploration_rate-self.exploration_decrease
		if(self.pretrain):
			return
		self.print_details()
		if(self.number_episodes%10 == 0 and self.pretrain == False):
			self.experience_replay()

	def experience_replay(self):
		self.train_df = self.train_df.sort_index(ascending=False)
		for index, serie in self.train_df.iterrows():
			#Update Q Value
			old_q = serie["Q_Values"][serie["Action"]]
			new_q = (serie["Reward"])
			serie["Q_Values"][serie["Action"]] = round(self.alpha*new_q+(1-self.alpha)*old_q,4)

		x = p.DataFrame(self.train_df.Observation.values.tolist())
		y = p.DataFrame(self.train_df.Q_Values.values.tolist())
		self.model.train_model(x,y)
		self.train_df = p.DataFrame(columns=["Observation","Q_Values","Action","Reward","new_Observation"])

	def print_details(self):
		print('Episode: {}'.format(self.number_episodes),
		'Total reward: {}'.format(self.episode_interations-1),
		'Explore P: {:.4f}'.format(self.exploration_rate))

	def save_machine(self):
		self.model.save_model()
