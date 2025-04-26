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

import carb
from omni.isaac.core import World
from omni.isaac.franka import KinematicsSolver
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.franka.tasks import FollowTarget
import numpy as np
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.cortex.cortex_utils import get_assets_root_path_or_die
from typing import Union
from pxr import Sdf, Usd, UsdGeom
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.stage import add_reference_to_stage

class AGVDriveStraightTask(BaseTask):
    def __init__(self, speed=0.02 , name="agv_drive_straight"):
        super().__init__(name)
        self.speed = speed
        self.inispeed = speed
        self.rotation_speed = 1.0
        self.time_keeper = 0.0
        self.agv = None
        self.env_path = "/World"
        self.q = euler_to_quaternion(-180, 0, -180)
        self.x_pose = 3.35
        self.y_pose = 15
        self.pause_timer = 0.0  # 停顿计时器
        self.state = "INITIALIZING"
        
        # add stop time and pass to pause_agv
        self.path_points = [
            {"position": np.array([3.35, 15, -0.3]), "rotation": -180, "stoptime": 300},  # 起始位置和朝向
            {"position": np.array([3.35, 12.4, -0.3]), "rotation": -270, "stoptime": 300},  # 第1个停靠点，旋转90度
            {"position": np.array([0.75, 12.4, -0.3]), "rotation": -180, "stoptime": 300},  # 第2个停靠点，旋转-90度    ingredient shelf
            {"position": np.array([0.75, 9.2, -0.3]), "rotation": -180, "stoptime": 900},  # 第3个停靠点，旋转0度       machine center
            {"position": np.array([0.75, 2.85, -0.3]), "rotation": -180, "stoptime": 300},  # 第4个停靠点，旋转0度      table
            {"position": np.array([0.75, -4.9, -0.3]), "rotation": -90, "stoptime": 300},  # 第5个停靠点，旋转90度
            {"position": np.array([5, -4.9, -0.3]), "rotation": 0, "stoptime": 300},  # 第6个停靠点，旋转-90度
            {"position": np.array([5, 15, -0.3]), "rotation": 90, "stoptime": 300},  # 第二个停靠点，旋转90度
        ]
        self.current_point_index = 0  # 当前目标停靠点索引
        self.target_rotation = 0  # 目标旋转角度
        self.current_rotation = 0  # 当前旋转角度

    def pre_step(self, time_step_index, simulation_time):
        self.time_keeper += 1.0  # 统一更新时间步长

        if self.state == "INITIALIZING":
            self._initialize_agv()
            self._initialize_block()
        elif self.state == "MOVING":
            self._move_agv()

        elif self.state == "PAUSED":
            self._pause_agv()

        elif self.state == "RESUMING":
            self._resume_movement()

        elif self.state == "ROTATING":
            self._rotate_agv()

        else:
            self.speed = 0  # 未知状态下停止

    def _initialize_agv(self):
        """初始化AGV"""
        if self.agv is None:  # 仅初始化一次
            prim_path = self.env_path + "/agv"
            add_reference_to_stage(
                "/home/bohan/Desktop/PRP46_models/agv.usdc", prim_path=prim_path
            )
            agv_prim = XFormPrim(
                prim_path,
                position=np.array([self.x_pose, self.y_pose,-0.55]),
                orientation=self.q,
                scale=[0.0015, 0.0015, 0.0015],
            )
            self.agv = self.scene.add(agv_prim)
            
        self.state = "ROTATING"  # 初始化完成后进入移动状态

    def _move_agv(self):
        """AGV移动逻辑"""
        if self.current_point_index < len(self.path_points):
            current_point = self.path_points[self.current_point_index]
            target_position = current_point["position"]

            # 判断是否到达当前目标点
            if np.allclose([self.x_pose, self.y_pose], target_position[:2], atol=0.1):  
                if current_point["rotation"] != self.current_rotation:
                    # 如果当前位置的旋转角度与目标旋转角度不同，则需要旋转
                    self.state = "ROTATING"
                else:
                    self.state = "PAUSED"  # 停靠
            else:
                # 正常移动到下一个目标点
                direction = target_position[:2] - np.array([self.x_pose, self.y_pose])
                distance = np.linalg.norm(direction)
                if distance > self.speed:
                    move_direction = direction / distance
                    self.x_pose += move_direction[0] * self.speed
                    self.y_pose += move_direction[1] * self.speed
                else:
                    self.x_pose, self.y_pose = target_position[:2]
                self.agv.set_world_pose(position=np.array([self.x_pose, self.y_pose, -0.5]))
        else:
            self.speed = 0
            self.state = "STOPPED"  # 所有停靠点完成后停止

    def _pause_agv(self):
        """在停靠点停顿逻辑"""
        current_point = self.path_points[self.current_point_index]
        stoptime= current_point["stoptime"]
        self.speed = 0  # 停止移动
        self.pause_timer += 1.0  # 更新停顿时间
        if self.pause_timer > stoptime:  # 停顿30秒后恢复运动     set this parameter
            self.state = "RESUMING"
            self.pause_timer = 0.0  # 重置停顿计时器

    def _resume_movement(self):
        """恢复运动逻辑"""
        self.current_point_index += 1  # 更新到下一个停靠点
        self.speed = self.inispeed  # 恢复初始速度
        self.state = "MOVING"  # 切换回移动状态

    def _rotate_agv(self):
        """AGV缓慢旋转逻辑"""
        current_point = self.path_points[self.current_point_index]
        target_rotation = current_point["rotation"]
        
        # 计算当前旋转角度与目标角度的差值
        angle_diff = target_rotation - self.current_rotation
        
        # 判断旋转方向
        if abs(angle_diff) > self.rotation_speed:  # 如果旋转角度差大于旋转速度，继续旋转
            if angle_diff > 0:
                # 顺时针旋转
                self.current_rotation += self.rotation_speed
                if self.current_rotation > target_rotation:
                    self.current_rotation = target_rotation  # 确保不超过目标角度
            else:
                # 逆时针旋转
                self.current_rotation -= self.rotation_speed
                if self.current_rotation < target_rotation:
                    self.current_rotation = target_rotation  # 确保不超过目标角度
        else:
            # 旋转完成，进入停靠状态
            self.current_rotation = target_rotation
            self.state = "PAUSED"
        
        # 更新AGV朝向
        q = euler_to_quaternion(-180, 0, self.current_rotation)
        self.agv.set_world_pose(position=np.array([self.x_pose, self.y_pose, -0.5]), orientation=q)


    def start_rotation(self, target_angle):
        """开始旋转"""
        self.target_rotation = target_angle
        self.state = "ROTATING"
    
    def get_observations(self):
        observations = {
            "agv_position": np.array([self.x_pose, self.y_pose, -0.5]),
            "agv_state": self.state,
            "agv_speed": self.speed,
            "current_point_index": self.current_point_index  # 应该使用 current_point_index
        }
        return observations


# Initialize the world and tasks
my_world = World(stage_units_in_meters=1.0)

assets_root_path = get_assets_root_path_or_die()
add_reference_to_stage(usd_path=assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd", prim_path="/World/Background")    

ori=euler_to_quaternion(0,0,0)
background_prim = XFormPrim(
    "/World/Background", position=[0, 0, -0.01], orientation=ori
)
    
# set stationary objects
stationary_prim= XFormPrim(
    "/World/stationary", position= [0, 0, -0.01], orientation=ori, scale=[1,1,1] 
)      # have to first declare xform of outer prim , then load model

add_reference_to_stage("/home/bohan/Desktop/PRP46_models/shelf.usdc", prim_path="/World/stationary/ingredient_shelf1")
add_reference_to_stage("/home/bohan/Desktop/PRP46_models/shelf.usdc", prim_path="/World/stationary/ingredient_shelf2")
add_reference_to_stage("/home/bohan/Desktop/PRP46_models/shelf.usdc", prim_path="/World/stationary/product_shelf1")
add_reference_to_stage("/home/bohan/Desktop/PRP46_models/shelf.usdc", prim_path="/World/stationary/product_shelf2")
add_reference_to_stage("/home/bohan/Desktop/PRP46_models/machiningCenter.usdc", prim_path="/World/stationary/machine_center")
add_reference_to_stage("/home/bohan/Desktop/PRP46_models/thor_table.usd", prim_path="/World/stationary/table")

shelf1_prim= XFormPrim(
    "/World/stationary/ingredient_shelf1", translation=[-4.55,15,0], 
    orientation=euler_to_quaternion(90,0,0),scale= [0.01,0.01,0.01]
)
shelf2_prim= XFormPrim(
    "/World/stationary/ingredient_shelf2", translation=[-1.5,15,0], 
    orientation=euler_to_quaternion(90,0,0),scale= [0.01,0.01,0.01]
)
shelf3_prim= XFormPrim(
    "/World/stationary/product_shelf1", translation=[-4.55,-6,0], 
    orientation=euler_to_quaternion(90,0,0),scale= [0.01,0.01,0.01]
)
shelf2_prim= XFormPrim(
    "/World/stationary/product_shelf2", translation=[-1.5,-6,0], 
    orientation=euler_to_quaternion(90,0,0),scale= [0.01,0.01,0.01]
)
machine_prim= XFormPrim(
    "/World/stationary/machine_center", translation=[3.5,7.5,0], 
    orientation=euler_to_quaternion(180,0,-90), scale= [0.0015,0.0015,0.0015]
)
table_prim= XFormPrim(
    "/World/stationary/table", translation=[2.5,0,1.18], 
    orientation=euler_to_quaternion(0,0,0),scale= [1.3,8,1.5]
)
# end of stationary objects


my_task_1 = FollowTarget(name="follow_target_task_1")
my_task_2 = FollowTarget(name="follow_target_task_2")
my_world.add_task(my_task_1)
my_world.add_task(my_task_2)
my_world.add_task(AGVDriveStraightTask(speed=0.01))
my_world.reset()

# Get task parameters
task_params_1 = my_world.get_task("follow_target_task_1").get_params()
task_params_2 = my_world.get_task("follow_target_task_2").get_params()
franka_name_1 = task_params_1["robot_name"]["value"]
target_name_1 = task_params_1["target_name"]["value"]
franka_name_2 = task_params_2["robot_name"]["value"]
target_name_2 = task_params_2["target_name"]["value"]

# Get robot articulations and controllers
my_franka_1 = my_world.scene.get_object(franka_name_1)
my_franka_2 = my_world.scene.get_object(franka_name_2)
my_controller_1 = KinematicsSolver(my_franka_1)
my_controller_2 = KinematicsSolver(my_franka_2)
articulation_controller_1 = my_franka_1.get_articulation_controller()
articulation_controller_2 = my_franka_2.get_articulation_controller()

robot1_prim=XFormPrim(
    "/World/Franka", position=[-1.5, 0, 0], orientation=ori
)

robot2_prim=XFormPrim(
    "/World/Franka_1", position=[0, 0, 0], orientation=ori
)

ground_prim=XFormPrim(
    "/World/defaultGroundPlane", position=[0,0,-0.1]
)

# Set initial parameters for circular motion
radius = 0.5
a=0.5
b=0.3
angular_speed = 0.005  # radians per simulation step
angle1 = 0.0
angle2 = 0.0

# Set initial positions for the second robot arm and target cube
initial_position_2 = np.array([1.0, 1.0, 0.0])
my_franka_2.set_world_pose(position=initial_position_2)
my_world.scene.get_object(target_name_1).set_visibility(False)
my_world.scene.get_object(target_name_2).set_visibility(False)

ground_plane = my_world.scene.get_object("defaultGroundPlane") 
if ground_plane: 
    ground_plane.set_visibility(False) 
else: 
    carb.log_warn("Ground plane not found in the scene.")

reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            reset_needed = False
            ground_prim=XFormPrim("/World/defaultGroundPlane", position=[0,0,-0.1])
        
        angle1 += angular_speed
        angle2 -= angular_speed
        target_position_1 = np.array([a * np.cos(angle1), b * np.sin(angle1), 0.2])
        target_position_2 = np.array([radius * np.cos(angle2), radius * np.sin(angle2), 0.2])
        my_world.scene.get_object(target_name_1).set_world_pose(position=target_position_1)
        my_world.scene.get_object(target_name_2).set_world_pose(position=target_position_2)
        
        # Get observations and apply actions for both robot arms
        observations = my_world.get_observations()
        actions_1, succ_1 = my_controller_1.compute_inverse_kinematics(
            target_position=observations[target_name_1]["position"],
            target_orientation=observations[target_name_1]["orientation"],
        )
        actions_2, succ_2 = my_controller_2.compute_inverse_kinematics(
            target_position=observations[target_name_2]["position"],
            target_orientation=observations[target_name_2]["orientation"],
        )
        if succ_1:
            articulation_controller_1.apply_action(actions_1)
        else:
            carb.log_warn("IK did not converge to a solution for Robot Arm 1. No action is being taken.")
        if succ_2:
            articulation_controller_2.apply_action(actions_2)
        else:
            carb.log_warn("IK did not converge to a solution for Robot Arm 2. No action is being taken.")

simulation_app.close()