from robot import *
from game import *
from humanPolicy import *
from humannode import *
from root import *
from robotnode import *
from pomcp import *
import math

num_theta = 4
#num_theta = 6
horizon = 0
num_ingredients = 3
#num_ingredients = 5

robot_belief = [1/num_theta for i in range(num_theta)]
reward_set = [((0,2,1),0), ((1,0,2),1), ((2,1,1),2), ((2,0,2),3)]
#reward_set = [((0,4,8),0), ((4,0,8),1), ((8,2,2),2), ((2,2,6),3)]
#reward_set = [((2,0,4,0,2), 0), ((0,0,2,0,6), 1), ((0,0,3,5,0), 2), ((0,5,3,0,0), 3), ((4,0,4,0,0), 4), ((0,0,4,4,0), 5)]
initial_world_state = (0,0,0)
#initial_world_state = (0,0,0,0,0)
human_behavior = "boltzmann"


humanPolicy = HumanPolicy(num_actions = num_ingredients + 1, behavior = human_behavior)
robot = Robot(robot_belief, num_actions = num_ingredients + 1)
game = Game(robot, humanPolicy, initial_world_state, num_theta, num_ingredients, reward_set)

initial_history = Root(game, [((0,0,0),0), ((0,0,0),1), ((0,0,0),2), ((0,0,0),3)], 0)
#initial_history = Root(game, [((0,0,0),0), ((0,0,0),1), ((0,0,0),2), ((0,0,0),3)], 0)
#initial_history = Root(game, [((0,0,0,0,0),0), ((0,0,0,0,0),1), ((0,0,0,0,0),2), ((0,0,0,0,0),3), ((0,0,0,0,0),4), ((0,0,0,0,0),5)], 0)

#make sure to change exploration accordingly - also what should the epsilon value be?
epsilon = math.pow(0.95, 5)

# print("Required Horizon: 4")
# print("Number Of Theta: 6")
# print("Number Of Ingredients: 5")

for _ in range(0, 1):
#KEEP THESE PARAMETERS FOR NOW!!
	solver = POMCP_Solver(0.95, epsilon, 100000, initial_history, game, 300, 5)
	solver.search()
	print("_____________________")

"""
Things to keep in mind:
1. Make sure to change epsilon appropriately.
2. The game file has that weird shit about leaving the same state when the human and robot have the same
   action. Not sure if this matters or not? Maybe depends on epsilon?
3. Exploration should be >= 300 for >= million, maybe just a wee bit less for 10000.
"""