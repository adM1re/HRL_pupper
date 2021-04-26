import pybullet as p
import time
import pybullet_data

useDeepLocoCSV = 2
useAdmireCSV = 3
heightfieldSource = 0
physicsClient = p.connect(p.GUI)
# or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
# optionally
p.setGravity(0, 0, -9.8)
numHeightfieldRows = 16
numHeightfieldColumns = 16
show_pupper = 1
show_ground = 0
if heightfieldSource == useDeepLocoCSV:
  terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                        meshScale=[.5, .5, 2.5],
                                        fileName="heightmaps/ground0.txt",
                                        heightfieldTextureScaling=64)
  terrain = p.createMultiBody(0, terrainShape)
  terrain_position_start = [-7, 24, -1.8]
  terrain_position_mid = [-30, 30, -1.7]  # [30, -27~~-30, 1.7]
  terrain_position_end = [-50, 36, -1.7]  # [49~~52, -36, 1.7]  [42~~45, -12 , 0 ]

  terrain_orientation = [0, 0, 0, 1]
  p.resetBasePositionAndOrientation(terrain, terrain_position_start, terrain_orientation)
  p.changeVisualShape(terrain,
                      -1,
                      rgbaColor=[1, 1, 1, 1])
elif heightfieldSource == useAdmireCSV:
    terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                          meshScale=[.5, 0.5, 2.5],
                                          fileName="heightmaps/pupper_ground_test.txt",
                                          # heightfieldTextureScaling=128,
                                          numHeightfieldRows=80,
                                          numHeightfieldColumns=80)
    terrain = p.createMultiBody(baseCollisionShapeIndex=terrainShape)
    terrain_position_start = [-7, 24, -1.8]
    terrain_orientation = [0, 0, 0, 1]
    # p.resetBasePositionAndOrientation(terrain, terrain_position_start, terrain_orientation)
    p.changeVisualShape(terrain,
                        -1,
                        rgbaColor=[1, 1, 1, 1])
    print("yes")
if show_pupper:
    allbodyids = p.loadMJCF("pupper_pybullet_out.xml")
    bodyids = allbodyids[1]
    assert isinstance(bodyids, object)
    num_joints = p.getNumJoints(bodyids)
    pupper_position_start = [2, -3, 0.5]
    pupper_position_end = [42, -12, 0.5]
    pupper_orientation_start = [0, 0, 0, 1]
    p.resetBasePositionAndOrientation(bodyids, [2, 2, 0.5], [0, 0, 0, 1])
    pos = p.getBasePositionAndOrientation(bodyids)
    print(pos)
if show_ground:
    pupper_ground = p.loadURDF("ground_test.urdf")
    print("pupper_ground:")
    print(pupper_ground)
    p.resetBasePositionAndOrientation(pupper_ground, [-11, 0, -0.1], [1, 1, 1, 1])

# p.loadURDF("sphere_small.urdf")
# p.loadURDF("plane.urdf")
# p.loadURDF("table/table.urdf")
p.configureDebugVisualizer(p.GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
while (p.isConnected()):
    p.stepSimulation()
    # now_position, now_orientation = p.getBasePositionAndOrientation(bodyids)
    time.sleep(1./240.)
