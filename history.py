import numpy as np
import math

class HistoryNode:

	def __init__(self, key, action_type, belief, game, value, visited):
		"""
		Generates an instance of the HistoryNode class.
		:param head: stores the current action/observation
		:param children: is a dictionary which maps actions to future HistoryNodes
		:param action_type: stores whether the key is a robot or human action
		:param belief: is a list that stores all states visited during the search process
		"""
		self.key = key
		self.action_type = action_type
		self.children = self.create_children()
		self.belief = belief
		self.game = game
		self.value = value
		self.visited = visited
		self.augmented_value = 0 #so i can put the c in the POMCP solver

	def create_children(self):
		"""
		Creates the children for a particular history. Defaults all values in the dictionary to None.

		CHECK THE LOGIC HERE!!! I THINK IF CURRENT IS ROBOT THEN NEXT IS HUMAN THATS WHY I DID IT LIKE THIS!
		AND VICE VERSA

		DO A TEST TO MAKE SURE THAT THE CHILDREN AREN'T AN EMPTY DICTIONARY after creation

		"""
		new_dict = {}
		if self.action_type == "robot":
			for human_action in self.game.getAllObservations():
				new_dict[human_action] = None
		else if self.action_type == "human" or self.action_type == "None":
			for robot_action in self.game.getAllActions():
				new_dict[robot_action] = None
		else:
			print("LABELS ARE WRONG")

		return new_dict

	def check_rollout(self):
		"""
		Returns True if all future histories of children robot actions are None (ie all values
			in children dictionary are None), so a rollout should occur

		CHECK LOGIC OF MY ERROR TEST
		"""

		if self.action_type == "robot":
			print("SOMETHING IS WRONG")
			return

		return all(value == None for value in self.children.values())

	def find_child(self, action):
		"""
		:param action: the action who's child history we are looking for

		Return the history who's key is the action we want. Returns None if history doesn't exist.
		"""
		return self.children[action]

	def find_optimal_action(self, c):
		if not self.check_rollout():
			print("WTF THIS SHOULDNT HAPPEN")
			return

		for k, v in self.children.items():
			if v == None:
				return k


		list_to_sort = []
		for robot_action in self.children:
			robot_action_value = self.compute_augmented_value(c, robot_action)
			list_to_sort.append((robot_action, robot_action_value))

		list_to_sort = sorted(list_to_sort, key=lambda x: x[1], reverse = True)
		return list_to_sort[0][0]

	def sample_belief(self):
		"""
		Randomly samples and returns a state from the belief list of the history
		"""
		return np.random.choice(self.belief)

	def update_belief(self, state):
		"""
		Appends a state to the belief list of the history.
		"""
		self.belief.append(state)

	def increment_visited(self):
		self.visited = self.visited + 1

	def update_value(self, reward):
		self.value = self.value + ((reward - self.value)/self.visited)

	def compute_augmented_value(self, c, robot_action):
		augmented_value = value + c*math.pow((math.log(self.visited) / self.children[robot_action].visited), 0.5)
		return augmented_value

	def compute_boltzmann_probabilities(self, beta):
		"""
		:param beta: takes the beta parameter from the POMCP solver

		Returns an np array of probabilities of getting each human action, ordered by human actions

		CHECK MY NP MATH HERE
		"""
		list_of_probabilities = []
		qValues = []

		for human_action in self.game.getAllObservations():
			qValues.append(self.children[human_action].value)
		
		qValues = np.array(qValues)
		expQ_values = np.exp(beta * qValues)
		total = sum(expQ_values)

		list_of_probabilities = 1/total * qValues

		return np.array(list_of_probabilities)








