import numpy as np
import math

class POMCP_Solver:

	def __init__(self, gamma, epsilon, timer, history, game, c, beta):
		"""
		creates a instance of a POMCP_Solver.
		:param gamma: this is the discount factor
		:param epsilon: this is the tolerance factor at which the rollout can be stopped
		:param timer: this is how long the search function is called
		:param history: this is the history from which the search function will be called
		:param game: the game that we pass in
		:param c: the constant which affects how much exploration vs. exploitation we want
		:param beta: the constant which measures human rationality
		"""
		self.gamma = gamma
		self.epsilon = epsilon
		self.timer = timer
		self.history = history
		self.game = game
		self.c = c
		self.beta = beta

	def search(self):
		for _ in range(0, self.timer):
			sample_state = self.history.sample_belief()
			
			#self.simulate(sample_state, self.history, 0)

		#return argmax
		#self.history = history corresponding to argmax

	def rollout_first(self, state, depth):
		robot_action = np.random.choice(self.game.getAllActions())
		return self.rollout(state, depth, robot_action), robot_action


	def rollout(self, state, depth, next_robot_action):
		if math.pow(self.gamma, depth) < self.epsilon:
			return 0

		robot_action = np.random.choice(self.game.getAllActions())
		human_action = np.random.choice(self.game.getAllObservations())
		next_state = self.game.getNextState(state, robot_action, human_action)
		next_robot_action = np.random.choice(self.getAllActions())

		return self.game.getReward(next_state) + self.gamma*self.rollout(next_state, depth + 1, next_robot_action)

	def simulate(self, state, history, depth):
		if math.pow(self.gamma, depth) < self.epsilon:
			return 0
		
		#this para not be needed (taken care by internal if statement below)
		#but this could be helpful for debugging.
		# if history.check_rollout() == True and (history.key == "human_action" or history.key == "None"):
		# 	value, robot_action_append = self.rollout_first(state, depth)
		# 	new_history_node = HistoryNode(robot_action_append, "robot", [state], self.game, value, 1)
		# 	history[robot_action_append] = new_history_node
		# 	return

		if history.key == "human_action" or history.key == "None":
			optimal_action = history.find_optimal_action(self.c)
			
			if history[optimal_action] == None:
				value = self.rollout(state, depth, optimal_action)
				new_history_node = HistoryNode(optimal, "robot", [state], self.game, value, 1)
				history[optimal_action] = new_history_node
				return

			next_state, next_human_action, next_reward = self.sample_boltzmann(state, history, optimal_action)
			next_history = history.children[optimal_action].children[human_action]
			R = next_reward + self.gamma*self.simulate(next_state, next_history, depth + 1)
			self.belief.append(state)
			self.increment_visited()
			self.children[optimal_action].increment_visited()
			self.children[optimal_action].update_value(R)

			return

	def sample_boltzmann(self, state, history, robot_action):
		"""
		Samples a human action based on the future Q values.
		Returns the next state, future (sampled) human action, and reward.
		"""
		list_of_probabilities = history[robot_action].compute_boltzmann_probabilities(self.beta)
		human_actions = self.game.getAllObservations()
		sample_human_action = np.random.choice(human_actions, p = list_of_probabilities)

		next_state = self.game.getNextState(state, robot_action, sample_human_action)
		next_reward = self.game.getReward(next_state)

		return next_state, sample_human_action, next_reward






