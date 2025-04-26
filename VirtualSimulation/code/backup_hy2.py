#launch Isaac Sim before any other imports
#default first two lines in any standalone application
import math
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False}) # we can also run as headless.
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka import Franka
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.cortex.cortex_utils import get_assets_root_path_or_die
import numpy as np

my_world = World()

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
my_world.scene.add_default_ground_plane()

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
        self.paused = False
        
        # add stop time and pass to pause_agv
        self.path_points = [
            {"position": np.array([3.35, 15, -0.3]), "rotation": -180, "stoptime": 300},  # 起始位置和朝向
            {"position": np.array([3.35, 10.4, -0.3]), "rotation": -270, "stoptime": 300},  # 第1个停靠点，旋转90度
            {"position": np.array([0.75, 10.4, -0.3]), "rotation": -180, "stoptime": 300},  # 第2个停靠点，旋转-90度    ingredient shelf
            {"position": np.array([0.75, 7.8, -0.3]), "rotation": -180, "stoptime": 900},  # 第3个停靠点，旋转0度       machine center
            {"position": np.array([0.75, 2.5, -0.3]), "rotation": -180, "stoptime": 300},  # 第4个停靠点，旋转0度      table
            {"position": np.array([0.75, -4.9, -0.3]), "rotation": -90, "stoptime": 300},  # 第5个停靠点，旋转90度
            {"position": np.array([7.5, -4.9, -0.3]), "rotation": 0, "stoptime": 300},  # 第6个停靠点，旋转-90度
            {"position": np.array([7.5, 15, -0.3]), "rotation": 90, "stoptime": 300},  # 第二个停靠点，旋转90度
        ]
        self.current_point_index = 0  # 当前目标停靠点索引
        self.target_rotation = 0  # 目标旋转角度
        self.current_rotation = -180  # 当前旋转角度

    def pre_step(self, time_step_index, simulation_time):
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
            self.agv = self.scene.add(agv_prim)
            
        self.state = "ROTATING"  

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
my_world.add_task(AGVDriveStraightTask(speed=0.01, name="agv_drive_straight"))

# class cuberemove(BaseTask):
#     fancy_cube =  world.scene.add(
#         DynamicCuboid(
#             prim_path="/World/test/random_cube",
#             name="fancy_cube",
#             position=np.array([0, 0, 1.0]),
#             scale=np.array([0.5015, 0.5015, 0.5015]),
#             color=np.array([0, 0, 1.0]),
#         ))
#     franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", 
#                                     name="fancy_franka",
#                                     position=np.array([0, 0, 3.0])))
# Resetting the world needs to be called before querying anything related to an articulation specifically.
# Its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
my_world.reset()

while simulation_app.is_running():
    my_world.step(render=True) # execute one physics step and one rendering step

simulation_app.close() # close Isaac Sim