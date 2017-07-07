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
		"""
		The Search function as described in Silver et al.
		Returns the optimal action after the search tree has been constructed for timer timesteps.
		Also prunes the search tree so that it is now rooted at the optimal --robot-- action.
		
		NOTE: TO DO NEXT SEARCH, WE NEED TO RECEIVE A REAL OBSERVATION AND UPDATE THE ROOT AGAIN.
		ROOT MUST ALWAYS BE AT --HUMAN-- OR --NONE--.
		"""
		for _ in range(0, self.timer):
			sample_state = self.history.sample_belief()
			self.simulate(sample_state, self.history, 0)

		optimal_action = self.find_optimal_action_non_aug()
		self.history = self.history.children[optimal_action] #prune rest of search tree
		return optimal_action


	def rollout(self, state, depth, robot_action):
		"""
		This performs the rollout given an initial robot action. It selects randomly from a uniform
		distribution over robot and human actions.

		Returns the value of the rollout.
		"""
		if math.pow(self.gamma, depth) < self.epsilon:
			return 0

		human_action = np.random.choice(self.game.getAllObservations())
		next_state = self.game.getNextState(state, robot_action, human_action)
		next_robot_action = np.random.choice(self.getAllActions())

		return self.game.getReward(next_state) + self.gamma*self.rollout(next_state, depth + 1, next_robot_action)

	def simulate(self, state, history, depth):
		"""
		The Simulate function as described in Silver et al.
		Returns the reward received from moving through the search tree from an initial state and history.
		
		Most importantly, it constructs the search tree and updates the values of history nodes in
		the search tree.
		"""
		if math.pow(self.gamma, depth) < self.epsilon:
			return 0

		if history.key == "human_action" or history.key == "None":
			optimal_action = history.find_optimal_action(self.c)
			
			if history.children[optimal_action].visited == 0:
				value = self.rollout(state, depth, optimal_action)
				# new_history_node = HistoryNode(optimal_action, "robot", [state], self.game, value, 1)
				# new_history_node.create_children()
				# history[optimal_action] = new_history_node
				history.children[optimal_action].value = value
				history.children[optimal_action].increment_visited()
				history.children[optimal_action].update_belief(state)
				history.children[optimal_action].create_children()
				return

			next_state, next_human_action, next_reward = self.sample_boltzmann(state, history, optimal_action)
			next_history = history.children[optimal_action].children[next_human_action]
			if len(next_history.children) == 0:
				next_history.create_children()
			R = next_reward + self.gamma*self.simulate(next_state, next_history, depth + 1)
			self.belief.append(state)
			self.increment_visited()
			self.children[optimal_action].increment_visited()
			self.children[optimal_action].update_value(R)

			return R

	def sample_boltzmann(self, state, history, robot_action):
		"""
		Samples a human action based on the future Q values.
		Returns the next state, future (sampled) human action, and reward.

		CHECK NP MATH HERE.
		"""
		list_of_probabilities = history[robot_action].compute_boltzmann_probabilities(self.beta)
		human_actions = self.game.getAllObservations()
		sample_human_action = np.random.choice(human_actions, p = list_of_probabilities)

		next_state = self.game.getNextState(state, robot_action, sample_human_action)
		next_reward = self.game.getReward(next_state)

		return next_state, sample_human_action, next_reward






