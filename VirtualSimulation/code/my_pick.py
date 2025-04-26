import math

def euler_to_quaternion(roll1, pitch1, yaw1):
    roll=math.radians(roll1)
    pitch=math.radians(pitch1)
    yaw=math.radians(yaw1)
    half_roll = roll / 2
    half_pitch = pitch / 2
    half_yaw = yaw / 2
    sin_roll = math.sin(half_roll)
    cos_roll = math.cos(half_roll)
    sin_pitch = math.sin(half_pitch)
    cos_pitch = math.cos(half_pitch)
    sin_yaw = math.sin(half_yaw)
    cos_yaw = math.cos(half_yaw)
    w = cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw
    x = sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw
    y = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw
    z = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw

    return [w, x, y, z]

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
from omni.isaac.franka.tasks import PickPlace
from omni.isaac.core.prims.xform_prim import XFormPrim

my_world = World(stage_units_in_meters=1.0)
my_task = PickPlace()
my_world.add_task(my_task)
my_world.reset()
task_params = my_task.get_params()
my_franka = my_world.scene.get_object(task_params["robot_name"]["value"])

my_cube= my_world.scene.get_object(task_params["cube_name"]["value"])

my_controller = PickPlaceController(
    name="pick_place_controller", gripper=my_franka.gripper, robot_articulation=my_franka
)
articulation_controller = my_franka.get_articulation_controller()

# cube_prim=XFormPrim(
#     "/World/Cube", scale=[0.7,0.7,0.7]
# )


i = 0
reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            my_controller.reset()
            reset_needed = False
        observations = my_world.get_observations()
        actions = my_controller.forward(
            picking_position=observations[task_params["cube_name"]["value"]]["position"],
            # placing_position=observations[task_params["cube_name"]["value"]]["target_position"],
            placing_position=[0.5,0,0.5],
            current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
            end_effector_offset=np.array([0, 0.005 ,0]),
        )
        if my_controller.is_done():
            print("done picking and placing")
        articulation_controller.apply_action(actions)
simulation_app.close()
