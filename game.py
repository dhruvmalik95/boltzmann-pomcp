import itertools
from operator import *

class Game:
    """
    A model of the ChefWorld game, in its reduced POMDP form.

    Throughout, world_state denotes the true world state, an n-tuple where
    n = num_ingredients.

    State denotes the state in the POMDP, a tuple of the form
    (world_state, theta).
    """
    def __init__(self, robot, human_policy, initial_world_state,
            num_theta = 2, num_ingredients = 3, reward_set = [((0,2,1),0), ((1,1,2),1)],
            gamma = 0.95):
        """
        Initializes an instance of the Chefworld game.

        :param robot: the Robot player.
        :param human_policy: the policy that the human follows.
        :param initial_world_state: the initial world state.
        :param num_theta: the number of possible theta values.
        :param num_ingredients: the number of possible recipes in the game; note
            the number of human and robot actions is num_ingredients - 1.
        :param reward_set: the set of states in which
        """
        self.robot = robot
        self.human_policy = human_policy
        self.num_ingredients = num_ingredients
        self.reward_set = reward_set
        self.gamma = gamma
        self.world_state = initial_world_state
        self.theta_set = list(range(num_theta))
        self.allStates = self.getAllStates()
        self.allObservations = self.getAllObservations()

    # Functions which describe the underlying dynamics of the game.
    def getNextState(self, current_state, robot_action, human_action):
        """
        Returns the new state of the game after the robot (and possibly the
        human) acts.

        :param current_state: the current augmented state of the game.
        :param robot_action: the robot's action.
        :param human_action: the human's action.
        """
        # if robot_action == human_action:
        #     return current_state
        total_action = tuple(map(add, human_action, robot_action))
        state = (tuple(map(add, current_state[0], total_action)), current_state[1])
        return state

    def getReward(self, state):
        """
        Returns the reward when the game is in a particular state.

        :param state:
        """
        return (state in self.reward_set) * 1

    def getAllWorldStates(self):
        """
        Returns all possible world states in the game as a Python list.
        """
        arrays = []
        for i in range(self.num_ingredients):
            arrays.append(list(range(7))) #regular = 3, script_limit = 
        return list(itertools.product(*arrays))

    def getAllTheta(self):
        """
        Returns all possible values of theta as a Python list.
        """
        return self.theta_set

    def getAllStates(self):
        """
        Returns all possible states in the POMDP game as a Python list.
        """
        return list(itertools.product(self.getAllWorldStates(), self.getAllTheta()))

    def getAllActions(self):
        """
        Returns all possible robot actions in the game as a Python list.
        """
        return self.robot.actions

    def getAllObservations(self):
        """
        Returns all possible observations (i.e. human actions) in the game as a
        Python list.
        """
        return self.human_policy.actions

    def transition(self, initial_state, robot_plan, human_action, final_state):
        """
        Returns the probability of transitioning from initial_state to
        final_state given the robot's plan and the human's action.

        :param initial_state: the robot's current state.
        :param robot_plan: the robot's conditionalPlan.
        :param human_action: the human's action.
        :param final_state: the robot's final state.
        """
        return (self.getNextState(initial_state, robot_plan.action, human_action) == final_state) * 1
