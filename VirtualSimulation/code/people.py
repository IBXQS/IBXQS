import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.timeline
from omni.isaac.core.world import World
from omni.isaac.core.utils.extensions import enable_extension

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

        # Check the available assets for people
        people_assets_list = Person.get_character_asset_list()
        for person in people_assets_list:
            print(person)

        # Create a person without setting up a controller, and just setting a manual target position for it to track
        p2 = Person("person2", "original_female_adult_business_02", init_pos=[2.0, 0.0, 0.0])
        p2.update_target_position([10.0, 0.0, 0.0], 0.5)

        p3 = Person("person3", "original_female_adult_business_02", init_pos=[-2.0, 0.0, 0.0])
        p3.update_target_position([2.0, 0.0, 0.0], 0.5)
        p3.update_target_position([2.0, 4.0, 0.0], 0.5)

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:
            # Update the UI of the app and perform the physics step
            self.world.step(render=True)
            print("here111")
        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():

    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()

    print("here")

if __name__ == "__main__":
    main()