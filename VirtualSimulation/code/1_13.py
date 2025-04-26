import math
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
from omni.isaac.core.objects import DynamicCuboid
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
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
from omni.isaac.franka.tasks import PickPlace
from omni.isaac.core.objects import FixedCuboid

my_world = World(stage_units_in_meters=1.0)

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

class AGVDriveStraightTask():
    # def __init__(self, speed=0.02 , name="agv_drive_straight"):
    def __init__(self):
        speed=0.02
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
        self.paused = False
        print("initialized")
        
        # add stop time and pass to pause_agv
        self.path_points = [
            {"position": np.array([3.35, 15, -0.3]), "rotation": -180, "stoptime": 300},  # 起始位置和朝向
            {"position": np.array([3.35, 10.4, -0.3]), "rotation": -270, "stoptime": 30},  # 第1个停靠点，旋转90度
            {"position": np.array([0.75, 10.4, -0.3]), "rotation": -180, "stoptime": 150},  # 第2个停靠点，旋转-90度    ingredient shelf
            {"position": np.array([0.75, 7.8, -0.3]), "rotation": -180, "stoptime": 300},  # 第3个停靠点，旋转0度       machine center
            {"position": np.array([0.75, 2.5, -0.3]), "rotation": -180, "stoptime": 300},  # 第4个停靠点，旋转0度      table
            {"position": np.array([0.75, -4.9, -0.3]), "rotation": -90, "stoptime": 150},  # 第5个停靠点，旋转90度
            {"position": np.array([7.5, -4.9, -0.3]), "rotation": 0, "stoptime": 30},  # 第6个停靠点，旋转-90度
            {"position": np.array([7.5, 15, -0.3]), "rotation": 90, "stoptime": 300},  # 第二个停靠点，旋转90度
        ]
        self.current_point_index = 0  # 当前目标停靠点索引
        self.target_rotation = 0  # 目标旋转角度
        self.current_rotation = -180  # 当前旋转角度

    def pre_step(self):
        """执行每个时间步的任务逻辑"""
        self.time_keeper += 1.0  # 统一更新时间步长

        # 检查是否处于暂停状态，只有在播放状态下才进行任务更新
        if my_world.is_stopped():
            self.speed = 0  # 暂停时停止移动
            return

        if self.state == "INITIALIZING":
            self._initialize_agv()

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
            # self.agv = self.scene.add(agv_prim)
            
        self.state = "ROTATING"  

    def _reset_agv(self):
        prim_path = self.env_path + "/agv"
    
        agv_prim = XFormPrim(
            prim_path,
            position=np.array([3.35, 15,-0.55]),
            orientation=euler_to_quaternion(-180,0,-180),
            scale=[0.0015, 0.0015, 0.0015],
        )
            
        self.state = "ROTATING" 

        self.x_pose=3.35
        self.y_pose=15
        self.q=euler_to_quaternion(-180,0,180)
        self.current_point_index = 0
        self.target_rotation = 0
        self.current_rotation = -180

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

    def set_paused(self, paused: bool):
        self.paused = paused
    
    def get_observations(self):
        observations = {
            "agv_position": np.array([self.x_pose, self.y_pose, -0.5]),
            "agv_state": self.state,
            "agv_speed": self.speed,
            "current_point_index": self.current_point_index  # 应该使用 current_point_index
        }
        return observations


# Initialize the world and tasks

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
add_reference_to_stage("/home/bohan/Desktop/PRP46_models/people.usd",prim_path="/World/stationary/people")
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
people_prim = XFormPrim(
    "/World/stationary/people", translation=[3.7,2,0], 
    orientation=euler_to_quaternion(0,0,270),scale= [1,1,1]
)
# end of stationary objects

robot_position= [[2.5,2.6,1.15],[1.8,8.1,1.38]]
my_task_1 = PickPlace(name="pick_place_task_1", offset= robot_position[0])
my_world.add_task(my_task_1)
my_task_2 = PickPlace(name="pick_place_task_2", offset= robot_position[1])
my_world.add_task(my_task_2)

# add the agv to scene
my_agv=AGVDriveStraightTask()
my_agv._initialize_agv()
agv_prim = XFormPrim(
                "/World/agv",
                position=np.array([3.35, 15,-0.55]),
                orientation=euler_to_quaternion(-180, 0, -180),
                scale=[0.0015, 0.0015, 0.0015],
            )
my_agv.agv = my_world.scene.add(agv_prim)


my_world.reset()

task_params_1 = my_task_1.get_params()
task_params_2 = my_task_2.get_params()
my_franka_1 = my_world.scene.get_object(task_params_1["robot_name"]["value"])
my_franka_2 = my_world.scene.get_object(task_params_2["robot_name"]["value"])
target_name_1 = task_params_1["cube_name"]["value"]
target_name_2 = task_params_2["cube_name"]["value"]
cube_1=my_world.scene.get_object(target_name_1)
cube_2=my_world.scene.get_object(target_name_2)
cube_1.set_world_pose(position=[2.37, 3.1, 1.1])
cube_2.set_world_pose(position=[1.8,8.6,1.45])
my_controller_1 = PickPlaceController(
    name="pick_place_controller_1", gripper=my_franka_1.gripper, robot_articulation=my_franka_1, end_effector_initial_height=1.3
)
my_controller_2 = PickPlaceController(
    name="pick_place_controller_2", gripper=my_franka_2.gripper, robot_articulation=my_franka_2,end_effector_initial_height=1.5
)
articulation_controller_1 = my_franka_1.get_articulation_controller()
articulation_controller_2 = my_franka_2.get_articulation_controller()


ground_prim=XFormPrim(
    "/World/defaultGroundPlane", position=[0,0,-0.1]
)

block_table = my_world.scene.add(
            FixedCuboid(
                prim_path="/World/stationary/block_table", 
                name="block_spawn_table", 
                translation=np.array([2.37, 3.1, 1.1]),
                scale=np.array([0.2,0.2,0.1]),
                color=np.array([0,0,0]), 
            ))

arm_machine = my_world.scene.add(
            FixedCuboid(
                prim_path="/World/stationary/arm_machine", 
                name="arm_base_machine", 
                translation=np.array([1.8, 8.1, 1.35]),
                scale=np.array([0.3,0.3,0.1]),
                color=np.array([0.8,0.8,0.8]), 
            ))

block_machine = my_world.scene.add(
            FixedCuboid(
                prim_path="/World/stationary/block_machine", 
                name="block_spawn_machine", 
                translation=np.array([1.8, 8.6, 1.35]),
                scale=np.array([0.2,0.2,0.1]),
                color=np.array([0.8,0.8,0.8]), 
            ))
ingredient_cube1 =  my_world.scene.add(
        DynamicCuboid(
            prim_path="/World/ingredient_cube1",
            name="ingredient_cube1",
            position=np.array([0, 0, 1.0]),
            scale=np.array([0.1, 0.1, 0.1]),
            color=np.array([0, 0, 1.0]),
        ))
ingredient_cube2 =  my_world.scene.add(
        DynamicCuboid(
            prim_path="/World/ingredient_cube2",
            name="ingredient_cube2",
            position=np.array([0, 1, 1.0]),
            scale=np.array([0.1, 0.1, 0.1]),
            color=np.array([0, 0, 1.0]),
        ))
ingredient_cube3 =  my_world.scene.add(
        DynamicCuboid(
            prim_path="/World/ingredient_cube3",
            name="ingredient_cube3",
            position=np.array([0, 2, 1.0]),
            scale=np.array([0.1, 0.1, 0.1]),
            color=np.array([0, 0, 1.0]),
        ))

destination_1 = np.array([[2.37, 3.1, 1.2],[2.8,2.5,1.2]])
destination_2 = np.array([[1.8, 8.6, 1.45],[1.8, 8.6, 1.45]])

# declare global variables
print_iter=1
ingredient=10
product=0
machine_in_progress=0
manual_check=0
display_flag=0
# end of global variables

iter_1=0
iter_2=0
i1=0
i2=0
cube_spawned_machine=0
reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            my_controller_1.reset()
            my_controller_2.reset()
            reset_needed = False
            ground_prim=XFormPrim("/World/defaultGroundPlane", position=[0,0,-0.1])
            cube_1.set_world_pose(position=[2.37, 3.1, 1.1])
            cube_2.set_world_pose(position=[1.8,8.6,1.45])
            my_agv._reset_agv()
            # my_agv.agv.set_world_pose(position=np.array([3.35, 15, -0.5]), orientation=euler_to_quaternion(-180,0,-180))
            # my_agv.x_pose=3.35
            # my_agv.y_pose=15
            # my_agv.q=euler_to_quaternion(-180,0,180)
            # my_agv.state="ROTATING"
            
        my_agv.pre_step()
        if iter_1>30:
            observations = my_world.get_observations()
            actions = my_controller_1.forward(
                picking_position=observations[task_params_1["cube_name"]["value"]]["position"],
                placing_position=destination_1[i1%2],
                current_joint_positions=observations[task_params_1["robot_name"]["value"]]["joint_positions"],
                end_effector_offset=np.array([0, 0, 0]),
            )
            if my_controller_1.is_done():
                my_controller_1.reset()
                iter_1=0
                i1=i1+1
                # print(i)
            articulation_controller_1.apply_action(actions)
        iter_1=iter_1+1

        if iter_2>45:
            observations = my_world.get_observations()
            actions = my_controller_2.forward(
                picking_position=observations[task_params_2["cube_name"]["value"]]["position"],
                placing_position=destination_2[i2%2],
                current_joint_positions=observations[task_params_2["robot_name"]["value"]]["joint_positions"],
                end_effector_offset=np.array([0, 0, 0]),
            )
            # print(observations[task_params_2["cube_name"]["value"]]["position"])
            if my_controller_2.is_done():
                my_controller_2.reset()
                iter_2=0
                i2=i2+1
                # print(i)
            articulation_controller_2.apply_action(actions)
        iter_2=iter_2+1
        display_flag=display_flag+1
        if display_flag==500:
            print("==============================")
            print("Time: %d"%print_iter)
            print("ingredient: %d"%ingredient)
            print("Machining in progress: %d"%machine_in_progress)
            print("Manual check: %d"%manual_check)
            print("Product: %d"%product)
            print()
            print("==============================")
            display_flag=0           
            print_iter=print_iter+1
simulation_app.close()