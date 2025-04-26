import math
import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.timeline
from omni.isaac.core.world import World
from omni.isaac.core.utils.extensions import enable_extension


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

EXTENSIONS_PEOPLE = [
    'omni.anim.people', 
    'omni.anim.navigation.bundle', 
    'omni.anim.timeline',
    'omni.anim.graph.bundle', 
    'omni.anim.graph.core', 
    'omni.anim.graph.ui',
    'omni.anim.retarget.bundle', 
    'omni.anim.retarget.core',
    'omni.anim.retarget.ui', 
    'omni.kit.scripting',
    'omni.graph.io',
    'omni.anim.curve.core',
]

for ext_people in EXTENSIONS_PEOPLE:
    enable_extension(ext_people)

simulation_app.update()

import omni.usd
omni.usd.get_context().new_stage()

import numpy as np

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.people.person import Person
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface


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
        self.q = euler_to_quaternion(-180, 0, 90)
        self.x_pose = 3.35
        self.y_pose = 9.5
        self.pause_timer = 0.0  # 停顿计时器
        self.state = "INITIALIZING"
        self.paused = False
        self.carry = True
        self.count=0
        self.path_points = [
            {"position": np.array([3.35, 9.5, -0.32]), "rotation": -270, "stoptime": 30},  # start-down
            {"position": np.array([1.8, 9.5, -0.32]), "rotation": -180, "stoptime": 30},   # turn -right
            {"position": np.array([1.8, 7.5, -0.32]), "rotation": -180, "stoptime": 300},  # machine center -right
            {"position": np.array([1.8, 1.6, -0.32]), "rotation": -180, "stoptime": 300},  # table - right
            {"position": np.array([1.8, -4.5, -0.32]), "rotation": -90, "stoptime": 30},   # second turn - up
            {"position": np.array([5.4, -4.5, -0.32]), "rotation": 0, "stoptime": 30},     # turn - left
            {"position": np.array([5.4, 9.5, -0.32]), "rotation": -270, "stoptime": 30},   # turn - down
            {"position": np.array([3.35, 9.5, -0.32]), "rotation": -270, "stoptime": 30}   # end - down
        ]
        self.current_point_index = 0  # 当前目标停靠点索引
        self.target_rotation = 0  # 目标旋转角度
        self.current_rotation = -270  # 当前旋转角度

    def pre_step(self):
        """执行每个时间步的任务逻辑"""
        self.time_keeper += 1.0  # 统一更新时间步长

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
        num = input("Enter loop number: ")
        self.loopnum=int(num)
        if self.agv is None:  # 仅初始化一次
            prim_path = self.env_path + "/agv"
            add_reference_to_stage(
                "/home/bohan/Desktop/PRP46_models/agv.usdc", prim_path=prim_path
            )
            agv_prim = XFormPrim(
                prim_path,
                position=np.array([self.x_pose, self.y_pose,-0.32]),
                orientation=self.q,
                scale=[0.0011, 0.0011, 0.0011],
            )
            # self.agv = self.scene.add(agv_prim)
            
        self.state = "ROTATING"  

    def _reset_agv(self):
        prim_path = self.env_path + "/agv"
    
        agv_prim = XFormPrim(
            prim_path,
            position=np.array([3.35, 15,-0.32]),
            orientation=self.q,
            scale=[0.0011, 0.0011, 0.0011],
        )
            
        self.state = "ROTATING" 

        self.x_pose=3.35
        self.y_pose=9.5
        self.current_point_index = 0
        self.target_rotation = 0
        self.current_rotation = -270
        self.count=0

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
                self.agv.set_world_pose(position=np.array([self.x_pose, self.y_pose, -0.32]))
        else:
            self.current_point_index = 0    #back to initial position
            self.state = "PAUSED"   
            self.count=self.count+1
            if (self.count==self.loopnum):
                print("AGV stops.")
                self.state="STOPPED"
    def _pause_agv(self):
        """在停靠点停顿逻辑"""
        current_point = self.path_points[self.current_point_index]
        stoptime= current_point["stoptime"]
        self.speed = 0  # 停止移动
        self.pause_timer += 1.0  # 更新停顿时间
        self.paused=True
        if self.pause_timer > stoptime:  # 停顿30秒后恢复运动     set this parameter
            self.state = "RESUMING"
            self.pause_timer = 0.0  # 重置停顿计时器

    def _resume_movement(self):
        """恢复运动逻辑"""
        self.current_point_index += 1  # 更新到下一个停靠点
        self.speed = self.inispeed  # 恢复初始速度
        self.paused=False
        self.carry=True
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
        self.agv.set_world_pose(position=np.array([self.x_pose, self.y_pose, -0.32]), orientation=q)


    def start_rotation(self, target_angle):
        """开始旋转"""
        self.target_rotation = target_angle
        self.state = "ROTATING"

    def get_observations(self):
        observations = {
            "agv_position": np.array([self.x_pose, self.y_pose, -0.32]),
            "agv_state": self.state,
            "agv_speed": self.speed,
            "current_point_index": self.current_point_index  # 应该使用 current_point_index
        }
        return observations



class PegasusApp:
    def __init__(self):
        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics,
        # spawning asset primitives, etc.
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch one of the worlds provided by NVIDIA
        #self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
        self.pg.load_asset(SIMULATION_ENVIRONMENTS["Curved Gridroom"], "/World/layout")

    #start of stationanry models
        assets_root_path = get_assets_root_path_or_die()
        add_reference_to_stage(usd_path=assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd", prim_path="/World/Background")    

        ori=euler_to_quaternion(0,0,0)
        background_prim = XFormPrim(
            "/World/Background", position=[0, 0, -0.01], orientation=ori, scale=[0.75,0.75,0.75]
        )
            
        stationary_prim= XFormPrim(
            "/World/stationary", position= [0, 0, -0.01], orientation=ori, scale=[1,1,1] 
        )      # have to first declare xform of outer prim , then load model

        add_reference_to_stage("/home/bohan/Desktop/PRP46_models/shelf.usdc", prim_path="/World/stationary/ingredient_shelf1")
        add_reference_to_stage("/home/bohan/Desktop/PRP46_models/shelf.usdc", prim_path="/World/stationary/ingredient_shelf2")
        add_reference_to_stage("/home/bohan/Desktop/PRP46_models/shelf.usdc", prim_path="/World/stationary/product_shelf1")
        add_reference_to_stage("/home/bohan/Desktop/PRP46_models/shelf.usdc", prim_path="/World/stationary/product_shelf2")
        add_reference_to_stage("/home/bohan/Desktop/PRP46_models/machiningCenter.usdc", prim_path="/World/stationary/machine_center")
        add_reference_to_stage("/home/bohan/Desktop/PRP46_models/thor_table.usd", prim_path="/World/stationary/table")
        # add_reference_to_stage("/home/bohan/Desktop/PRP46_models/people.usd",prim_path="/World/stationary/people")
        shelf1_prim= XFormPrim(
            "/World/stationary/ingredient_shelf1", translation=[-4.55,12.5,0], 
            orientation=euler_to_quaternion(90,0,0),scale= [0.0075,0.0075,0.0075]
        )
        shelf2_prim= XFormPrim(
            "/World/stationary/ingredient_shelf2", translation=[-1.5,12.5,0], 
            orientation=euler_to_quaternion(90,0,0),scale= [0.0075,0.0075,0.0075]
        )
        shelf3_prim= XFormPrim(
            "/World/stationary/product_shelf1", translation=[-4.55,-6,0], 
            orientation=euler_to_quaternion(90,0,0),scale= [0.0075,0.0075,0.0075]
        )
        shelf4_prim= XFormPrim(
            "/World/stationary/product_shelf2", translation=[-1.5,-6,0], 
            orientation=euler_to_quaternion(90,0,0),scale= [0.0075,0.0075,0.0075]
        )
        machine_prim= XFormPrim(
            "/World/stationary/machine_center", translation=[3.5,7.5,0], 
            orientation=euler_to_quaternion(180,0,-90), scale= [0.001,0.001,0.001]
        )
        table_prim= XFormPrim(
            "/World/stationary/table", translation=[2.5,0,0.9], #-0.28
            orientation=euler_to_quaternion(0,0,0),scale= [0.975,6,1.125]
        )
        # end of stationary objects

        robot_position= [[2.6,1.8,0.87],[2.20418, 7.6818,0.89045]]
        my_task_1 = PickPlace(name="pick_place_task_1", offset= robot_position[0])
        self.world.add_task(my_task_1)
        my_task_2 = PickPlace(name="pick_place_task_2", offset= robot_position[1])
        self.world.add_task(my_task_2)

        # add the agv to scene
        self.my_agv=AGVDriveStraightTask()
        self.my_agv._initialize_agv()
        agv_prim = XFormPrim(
                        "/World/agv",
                        position=np.array([3.35, 15,-0.32]),
                        orientation=euler_to_quaternion(-180, 0, -180),
                        scale=[0.001, 0.001, 0.001],
                    )
        self.my_agv.agv = self.world.scene.add(agv_prim)

        self.world.reset()

        self.task_params_1 = my_task_1.get_params()
        self.task_params_2 = my_task_2.get_params()
        my_franka_1 = self.world.scene.get_object(self.task_params_1["robot_name"]["value"])
        my_franka_2 = self.world.scene.get_object(self.task_params_2["robot_name"]["value"])
        target_name_1 = self.task_params_1["cube_name"]["value"]
        target_name_2 = self.task_params_2["cube_name"]["value"]
        self.cube_1=self.world.scene.get_object(target_name_1)
        self.cube_2=self.world.scene.get_object(target_name_2)
        self.cube_1.set_world_pose(position=[2.43265, 2.32731, 0.9])  #on table
        self.cube_2.set_world_pose(position=[2.28722, 8.09372, 1.0])  #on machine
        self.my_controller_1 = PickPlaceController(
            name="pick_place_controller_1", gripper=my_franka_1.gripper, robot_articulation=my_franka_1, end_effector_initial_height=1.1
        )
        self.my_controller_2 = PickPlaceController(
            name="pick_place_controller_2", gripper=my_franka_2.gripper, robot_articulation=my_franka_2,end_effector_initial_height=1.0
        )
        self.articulation_controller_1 = my_franka_1.get_articulation_controller()
        self.articulation_controller_2 = my_franka_2.get_articulation_controller()
        self.arm1_pick=False
        self.arm2_pick=False

        ground_prim=XFormPrim(
            "/World/defaultGroundPlane", translation=np.array([0,0,-1])
        )
        block_table = self.world.scene.add(
                    FixedCuboid(
                        prim_path="/World/stationary/block_table", 
                        name="block_spawn_table", 
                        translation=np.array([2.43265, 2.32731, 0.84985]),
                        scale=np.array([0.2,0.2,0.05]),
                        color=np.array([0,0,0]), 
                    ))
        arm_machine = self.world.scene.add(
                    FixedCuboid(
                        prim_path="/World/stationary/arm_machine", 
                        name="arm_base_machine", 
                        translation=np.array([2.20418, 7.6818, 0.45354]),
                        scale=np.array([0.35,0.35,0.9]),
                        color=np.array([0.8,0.8,0.8]), 
                    ))
        block_machine = self.world.scene.add(
                    FixedCuboid(
                        prim_path="/World/stationary/block_machine", 
                        name="block_spawn_machine", 
                        translation=np.array([2.3, 8.17578, 0.91832]),
                        scale=np.array([0.15,0.3,0.03]),
                        color=np.array([0.8,0.8,0.8]), 
                    ))
        block_machine = self.world.scene.add(
                    FixedCuboid(
                        prim_path="/World/stationary/block_machine_2", 
                        name="block_spawn_machine_2", 
                        translation=np.array([2.63475, 7.52693, 0.91832]),
                        scale=np.array([0.15,0.3,0.03]),
                        color=np.array([0.8,0.8,0.8]), 
                    ))
        self.world.scene.get_object("block_spawn_machine_2").set_visibility(False)


        # self.destination_1 = np.array([[2.95933,1.8224,1.0],[2.43265, 2.32731, 1.0]])
        # self.destination_2 = np.array([[2.63475, 7.52693, 1.0],[2.28722, 8.09372, 1.0]])
        self.destination_1 = np.array([[2.95933,1.8224,1.0]])
        self.destination_2 = np.array([[2.63475, 7.52693, 1.0]])

        XFormPrim("/World/layout",position=np.array([0,0,-1]))

        self.p2 = Person("person2", "original_female_adult_business_02", init_pos=[3.6, -1.0, 0.0])

        self.world.reset()
        self.stop_sim = False

        self.display_table=True
        self.display_machine=True

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """
        self.timeline.play()
        self.p2.update_target_position([3.6, 1.0, 0.0], 0.5)

        # declare global variables
        self.ingredient=10
        self.product=0
        self.inmachine=0
        self.ontable=0
        # end of global variables

        reset_needed = True

        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:
            # Update the UI of the app and perform the physics step
            self.world.step(render=True)
            ground_prim=XFormPrim("/World/defaultGroundPlane", position=[0,0,-1])
            if self.world.is_stopped() and not reset_needed:
                reset_needed = True
            if self.world.is_playing():
                if reset_needed:
                    self.world.reset()
                    self.my_controller_1.reset()
                    self.my_controller_2.reset()
                    reset_needed = False
                    ground_prim=XFormPrim("/World/defaultGroundPlane", position=[0,0,-1])
                    self.cube_1.set_world_pose(position=[2.43265, 2.32731, 0.9])
                    self.cube_2.set_world_pose(position=[2.28722, 8.09372, 1.0])
                    self.my_agv._reset_agv()
                    self.cube_1.set_visibility(False)
                    self.cube_2.set_visibility(False)
                    self.ingredient=10
                    self.product=0
                    self.inmachine=0
                    self.ontable=0

                    
                self.my_agv.pre_step()

               
                if self.my_agv.paused==True and self.my_agv.carry==True and self.my_agv.current_point_index==3:
                    self.arm1_pick=True
                    self.my_agv.carry=False
                    self.cube_1.set_visibility(True)

                if self.arm1_pick==True:
                    observations = self.world.get_observations()
                    actions = self.my_controller_1.forward(
                        picking_position=observations[self.task_params_1["cube_name"]["value"]]["position"],
                        placing_position=self.destination_1[0],
                        current_joint_positions=observations[self.task_params_1["robot_name"]["value"]]["joint_positions"],
                        end_effector_offset=np.array([0, 0, 0]),
                    )

                    if self.display_table==True:
                        self.display_table=False
                        self.inmachine-=1
                        self.ontable+=1
                        print("==============================")
                        print("Manual check")
                        print("ingredient: %d"%self.ingredient)
                        print("Machining in progress: %d"%self.inmachine)
                        print("Manual check: %d"%self.ontable)
                        print("Product: %d"%self.product)
                        print("==============================")

                    if self.my_controller_1.is_done():
                        self.my_controller_1.reset()
                        self.arm1_pick=False
                        self.cube_1.set_visibility(False)
                        self.cube_1.set_world_pose(position=[2.43265, 2.32731, 0.9])
                        self.display_table=True
                        
                    self.articulation_controller_1.apply_action(actions)

                if self.my_agv.paused==True and self.my_agv.current_point_index==1 and self.my_agv.carry==True:
                    self.my_agv.carry=False
                    self.ingredient-=1
                    print("==============================")
                    print("Take one ingredient")
                    print("ingredient: %d"%self.ingredient)
                    print("Machining in progress: %d"%self.inmachine)
                    print("Manual check: %d"%self.ontable)
                    print("Product: %d"%self.product)
                    print("==============================")

                if self.my_agv.paused==True and self.my_agv.carry==True and self.my_agv.current_point_index==2:            #if agv just arrive at the desk
                    self.arm2_pick=True
                    self.my_agv.carry=False
                    self.cube_2.set_visibility(True)

                if self.arm2_pick==True:  
                    observations = self.world.get_observations()
                    actions = self.my_controller_2.forward(
                        picking_position=observations[self.task_params_2["cube_name"]["value"]]["position"],
                        placing_position=self.destination_2[0],
                        current_joint_positions=observations[self.task_params_2["robot_name"]["value"]]["joint_positions"],
                        end_effector_offset=np.array([0, 0, 0]),
                    )
                    if self.display_machine==True:
                        self.display_machine=False
                        self.inmachine+=1
                        print("==============================")
                        print("Machine")
                        print("ingredient: %d"%self.ingredient)
                        print("Machining in progress: %d"%self.inmachine)
                        print("Manual check: %d"%self.ontable)
                        print("Product: %d"%self.product)
                        print("==============================")

                    if self.my_controller_2.is_done():
                        self.my_controller_2.reset()
                        self.arm2_pick=False
                        self.cube_2.set_visibility(False)
                        self.cube_2.set_world_pose(position=[2.28722, 8.09372, 1.0])
                        self.display_machine=True
                        
                    self.articulation_controller_2.apply_action(actions)


        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():
    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    main()