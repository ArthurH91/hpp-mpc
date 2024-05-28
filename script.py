from hpp.corbaserver import Client, Robot, ProblemSolver
from hpp.gepetto import ViewerFactory

import json 

Client().problem.resetProblem()

Robot.urdfFilename = "/home/arthur/Desktop/Code/manipulation-task/hpp-mpc/franka_manipulation/urdf/franka2.urdf"
Robot.srdfFilename = "/home/arthur/Desktop/Code/manipulation-task/hpp-mpc/franka_manipulation/srdf/franka2.srdf"
robot = Robot("panda", rootJointType="anchor")
ps = ProblemSolver(robot)

vf = ViewerFactory(ps)
vf.loadObstacleModel("/home/arthur/Desktop/Code/manipulation-task/hpp-mpc/franka_manipulation/urdf/big_box.urdf", "box")
for i in range(5):
    name = f'box/base_link_{i}'
    pos = ps.getObstaclePosition(name)
    pos[2] += .8
    vf.moveObstacle(name, pos)
v = vf.createViewer(collisionURDF=True)

q_init = [0.70, 0.2820479965209959, 0.05794600009918227, -2.351319994330406, -0.794600009918227, 2.0937000320106747, 0.0, 0.0, 0.0]
q_goal = [-0.46356800079345684, 0.2820479965209959, 0.05794600009918227, -2.351319994330406, -0.794600009918227, 2.0937000320106747, 0.0, 0.0, 0.0]

v(q_init)

ps.selectPathPlanner("BiRRT*")
ps.setMaxIterPathPlanning(100)
ps.setInitialConfig(q_init)
ps.addGoalConfig(q_goal)

ps.solve()

# ps.getAvailable("pathoptimizer")
# ps.addPathOptimizer("RandomShortcut")
# ps.solve()
# ## If there are 2 optimized trajectories
path_length = ps.pathLength(0)

# print(f"for the trajectory non optimized, here's the configuration at the timestep 1 out of {path_length} : {ps.configAtParam(2,1)}")

# Another possibility:

# wp, times = ps.getWaypoints(2)


N = 21 # Number of nodes

X = [ps.configAtParam(0,i * path_length/N)[:7] for i in range(N)]
print(X)
with open(
    "results/warmstart"
    + ".json",
    "w",
) as outfile:
    json.dump(X, outfile)