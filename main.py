import time
import numpy as np
import pinocchio as pin
import json, codecs

from os.path import join, dirname, abspath
from wrapper_meshcat import MeshcatWrapper
from wrapper_panda import PandaWrapper
from ocp import OCPPandaReachingColWithMultipleCol
from scenes import Scene

### PARAMETERS
# Number of nodes of the trajectory
T = 20
# Time step between each node
dt = 0.01

# Creating the robot
robot_wrapper = PandaWrapper(capsule=True, auto_col=True)
rmodel, cmodel, vmodel = robot_wrapper()

# Creating the scene
scene = Scene()
cmodel, TARGET, q0 = scene.create_scene(rmodel, cmodel, "box")

# Generating the meshcat visualizer
MeshcatVis = MeshcatWrapper()
vis, meshcatVis = MeshcatVis.visualize(
    TARGET,
    robot_model=rmodel,
    robot_collision_model=cmodel,
    robot_visual_model=vmodel,
)

#### WARMSTART 
curr_path = dirname(str(abspath(__file__)))
name_warmstart_file = "warmstart.json"
path = join(join(curr_path, "results"), name_warmstart_file)
json_file = codecs.open(
        path, "r", encoding="utf-8"
).read()
# Loading the json file
traj_warmstart = json.loads(json_file)
XS_init = []
for q in traj_warmstart:
    x = np.concatenate((np.array(q), np.zeros(rmodel.nv) ))
    XS_init.append(x)
X = [q + [0] * rmodel.nv for q in traj_warmstart]
XS_init = [np.array(q + [0] * rmodel.nv) for q in traj_warmstart]
### INITIAL X0
x0 = np.concatenate([q0, pin.utils.zero(rmodel.nv)])

### CREATING THE PROBLEM WITHOUT WARM START
problem = OCPPandaReachingColWithMultipleCol(
    rmodel,
    cmodel,
    TARGET,
    T,
    dt,
    x0,
    WEIGHT_GRIPPER_POSE=100,
    WEIGHT_xREG=1e-2,
    WEIGHT_uREG=1e-4,
    SAFETY_THRESHOLD=1e-2,
    
)
ddp = problem()

# XS_init = [x0] * (T+1)
# US_init = [np.zeros(rmodel.nv)] * T
US_init = ddp.problem.quasiStatic(XS_init[:-1])

# Solving the problem
ddp.solve(XS_init, US_init)

print("End of the computation, press enter to display the traj if requested.")
### DISPLAYING THE TRAJ
while True:
    vis.display(q0)
    input()
    for xs in ddp.xs:
        vis.display(np.array(xs[:7].tolist()))
        time.sleep(1e-1)
    input()
    print("replay")
    
