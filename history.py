import numpy as np
import math

class HistoryNode:

	def __init__(self, key, action_type, belief, game, value, visited):
		"""
		Generates an instance of the HistoryNode class.
		:param key: stores the current action/observation
		:param children: is a dictionary which maps actions to future HistoryNodes
		:param action_type: stores whether the key is a robot or human action
		:param belief: is a list that stores all states visited during the search process
		"""
		self.key = key
		self.action_type = action_type
		self.children = {}
		self.belief = belief
		self.game = game
		self.value = value
		self.visited = visited
		self.augmented_value = 0 #so i can put the c in the POMCP solver

	def create_children(self):
		"""
		Creates the children for a particular history. 
		"""
		
		if self.action_type == "robot":
			for human_action in self.game.getAllObservations():
				new_node = HistoryNode(human_action, "human", [], self.game, 0, 0)
				self.children[human_action] = new_node
		else if self.action_type == "human" or self.action_type == "None":
			for robot_action in self.game.getAllActions():
				new_node = HistoryNode(robot_action, "robot", [], self.game, 0, 0)
				self.children[robot_action] = new_node
		else:
			print("LABELS ARE WRONG")


	def find_child(self, action):
		"""
		:param action: the action who's child history we are looking for

		Return the history who's key is the action we want. Returns None if history doesn't exist.
		"""
		return self.children[action]

	def find_optimal_action(self, c):
		"""
		:param c: the constant controlling exploitation vs. exploration

		Returns the optimal action of a history who's key is human or None, based on augmented values 
		of children.
		"""

		for k, v in self.children.items():
			if v.visited == 0:
				return k

		list_to_sort = []
		for robot_action in self.children:
			robot_action_value = self.compute_augmented_value(c, robot_action)
			list_to_sort.append((robot_action, robot_action_value))

		list_to_sort = sorted(list_to_sort, key=lambda x: x[1], reverse = True)
		return list_to_sort[0][0]

	def find_optimal_action_non_aug(self):
		"""
		Returns the optimal action of a history who's key is human or None, based on regular values 
		of children.
		"""

		list_to_sort = []
		for robot_action in self.game.getAllActions():
			robot_action_value = self.children[robot_action].value
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
		"""
		Increments the number of times a history has been visited.
		"""
		self.visited = self.visited + 1

	def update_value(self, reward):
		"""
		:param reward: Takes the current value and averages it with the reward just received.
		"""
		self.value = self.value + ((reward - self.value)/self.visited)

	def compute_augmented_value(self, c, robot_action):
		"""
		:param c: the constant controlling exploitation vs. exploration
		:param robot_action: the robot action who's value we want to determine.

		Returns the augmented value of taking a particular robot action.
		"""
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
			if self.children[human_action].visited == 0:
				qValues.append(0)
			else:
				qValues.append(self.children[human_action].value)
		
		qValues = np.array(qValues)
		expQ_values = np.exp(beta * qValues)
		total = sum(expQ_values)

		list_of_probabilities = 1/total * qValues

		return np.array(list_of_probabilities)








