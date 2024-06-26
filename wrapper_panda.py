# 2-Clause BSD License

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from os.path import dirname, join, abspath

import numpy as np
import pinocchio as pin
import hppfcl

from wrapper_meshcat import RED

# This class is for unwrapping an URDF and converting it to a model. It is also possible to add objects in the model,
# such as a ball at a specific position.


class PandaWrapper:
    def __init__(
        self,
        auto_col=False,
        capsule=False,
    ):
        """Create a wrapper for the robot panda.

        Args:
            auto_col (bool, optional): Include the auto collision in the collision model. Defaults to False.
            capsule (bool, optional): Transform the spheres and cylinder of the robot into capsules. Defaults to False.
        """

        # Importing the model
        pinocchio_model_dir = join(
            dirname(((str(abspath(__file__))))), "models"
        )
        model_path = join(pinocchio_model_dir, "franka_description/robots")
        mesh_dir = pinocchio_model_dir
        urdf_filename = "franka2.urdf"
        urdf_model_path = join(join(model_path, "panda"), urdf_filename)
        srdf_model_path = model_path + "/panda/demo.srdf"
        self._urdf_model_path = urdf_model_path
        self._mesh_dir = mesh_dir
        self._srdf_model_path = srdf_model_path

        # Color of the robot
        self._color = np.array([249, 136, 126, 255]) / 255

        # Boolean describing whether the auto-collisions are in the collision model or not
        self._auto_col = auto_col

        # Transforming the robot from cylinders/spheres to capsules
        self._capsule = capsule

    def __call__(self):
        """Create a robot.

        Returns
        -------
        _rmodel
            Model of the robot
        _cmodel
            Collision model of the robot
        _vmodel
            Visual model of the robot


        """
        (
            self._rmodel,
            self._cmodel,
            self._vmodel,
        ) = pin.buildModelsFromUrdf(
            self._urdf_model_path, self._mesh_dir, pin.JointModelFreeFlyer()
        )

        q0 = pin.neutral(self._rmodel)

        jointsToLockIDs = [1, 9, 10]

        geom_models = [self._vmodel, self._cmodel]
        self._model_reduced, geometric_models_reduced = pin.buildReducedModel(
            self._rmodel,
            list_of_geom_models=geom_models,
            list_of_joints_to_lock=jointsToLockIDs,
            reference_configuration=q0,
        )

        self._vmodel_reduced, self._cmodel_reduced = (
            geometric_models_reduced[0],
            geometric_models_reduced[1],
        )
        
        if self._capsule:
            self.transform_model_into_capsules()

        if self._auto_col:
            self._cmodel_reduced.addAllCollisionPairs()
            # self._cmodel_reduced.addCollisionPair(pin.CollisionPair(self._cmodel_reduced.getGeometryId("panda2_link2_capsule37"),self._cmodel_reduced.getGeometryId("panda2_link6_capsule22") ))
            # self._cmodel_reduced.addCollisionPair(pin.CollisionPair(self._cmodel_reduced.getGeometryId("panda2_link4_capsule31"),self._cmodel_reduced.getGeometryId("panda2_link6_capsule22") ))
            # self._cmodel_reduced.addCollisionPair(pin.CollisionPair(self._cmodel_reduced.getGeometryId("panda2_link4_capsule31"),self._cmodel_reduced.getGeometryId("panda2_link5_capsule28") ))
            # self._cmodel_reduced.addCollisionPair(pin.CollisionPair(self._cmodel_reduced.getGeometryId("panda2_link4_capsule31"),self._cmodel_reduced.getGeometryId("panda2_link5_capsule25") ))
            # self._cmodel_reduced.addCollisionPair(pin.CollisionPair(self._cmodel_reduced.getGeometryId("panda2_link4_capsule31"),self._cmodel_reduced.getGeometryId("panda2_link3_capsule34") ))
            # self._cmodel_reduced.addCollisionPair(pin.CollisionPair(self._cmodel_reduced.getGeometryId("panda2_leftfinger_0"),self._cmodel_reduced.getGeometryId("support_link_0") ))

            pin.removeCollisionPairs(
                self._model_reduced, self._cmodel_reduced, self._srdf_model_path
            )
        
        
        # Modifying the collision model to add the capsules
        rdata = self._model_reduced.createData()
        cdata = self._cmodel_reduced.createData()
        q0 = pin.neutral(self._model_reduced)

        # Updating the models
        pin.framesForwardKinematics(self._model_reduced, rdata, q0)
        pin.updateGeometryPlacements(
            self._model_reduced, rdata, self._cmodel_reduced, cdata, q0
        )
                
        return (
            self._model_reduced,
            self._cmodel_reduced,
            self._vmodel_reduced,
        )
        
    def transform_model_into_capsules(self,):
        collision_model_reduced_copy = self._cmodel_reduced.copy()
        list_names_capsules = []

        for geom_object in collision_model_reduced_copy.geometryObjects:
            if isinstance(geom_object.geometry, hppfcl.Cylinder):
                if (geom_object.name[:-4] + "capsule_0") in list_names_capsules:
                    name = geom_object.name[:-4] + "capsule_" + "1"
                else:
                    name  = geom_object.name[:-4] + "capsule_" + "0"
                list_names_capsules.append(name)
                placement = geom_object.placement
                parentJoint = geom_object.parentJoint
                parentFrame = geom_object.parentFrame
                geometry =  geom_object.geometry
                geom = pin.GeometryObject(
                    name,
                    parentFrame,
                    parentJoint,
                    hppfcl.Capsule(geometry.radius, geometry.halfLength),
                    placement,
                )
                geom.meshColor = RED
                self._cmodel_reduced.addGeometryObject(geom)
                self._cmodel_reduced.removeGeometryObject(geom_object.name)
            elif isinstance(geom_object.geometry, hppfcl.Sphere) and "link" in geom_object.name:
                self._cmodel_reduced.removeGeometryObject(geom_object.name)



if __name__ == "__main__":
    from wrapper_meshcat import MeshcatWrapper

    # Creating the robot
    robot_wrapper = PandaWrapper(capsule=True, auto_col=True)
    rmodel, cmodel, vmodel = robot_wrapper()
    rdata = rmodel.createData()
    cdata = cmodel.createData()
    # Generating the meshcat visualizer
    MeshcatVis = MeshcatWrapper()
    vis = MeshcatVis.visualize(
        robot_model=rmodel, robot_visual_model=cmodel, robot_collision_model=cmodel
    )
    # vis[0].display(pin.randomConfiguration(rmodel))
    vis[0].display(np.array([0.5] * 7))

    pin.computeCollisions(rmodel, rdata, cmodel, cdata, pin.neutral(rmodel), False)
    for k in range(len(cmodel.collisionPairs)):
        cr = cdata.collisionResults[k]
        cp = cmodel.collisionPairs[k]
        print(
            "collision pair:",
            cmodel.geometryObjects[cp.first].name,
            ",",
            cmodel.geometryObjects[cp.second].name,
            "- collision:",
            "Yes" if cr.isCollision() else "No",
        )
    q = pin.randomConfiguration(rmodel)
    vis[0].display(pin.randomConfiguration(rmodel))