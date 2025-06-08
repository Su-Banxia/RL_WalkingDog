import pybullet as p
import pybullet_data
import numpy as np
import os

class RobotEnv:
    def __init__(self, render=False):
        # Initialize PyBullet
        if render:
            self.physicsClient = p.connect(p.GUI)  # Graphical mode
        else:
            self.physicsClient = p.connect(p.DIRECT)  # Headless mode
        
        # Set physics simulation parameters
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Load built-in PyBullet URDFs
        p.setGravity(0, 0, -9.8)  # Set gravity
        p.setTimeStep(1.0/240.0)  # Set simulation time step
        
        # Load plane
        self.planeId = p.loadURDF("plane.urdf")
        
        # Load robot model
        robot_urdf_path = "assets/dog.urdf"  
        self.robotId = p.loadURDF(robot_urdf_path, [0, 0, 0.60]) # Initial robot position
        
        # Get joint information
        self.num_joints = p.getNumJoints(self.robotId)
        self.joint_indices = []
        self.joint_names = []
        self.foot_joint_indices = [] 
        self.action_repeat = 2  # Number of times to repeat each action
        self.stagnation_counter = 0  # Stagnation counter

        # Control parameters
        self.max_steps = 2400
        self.target_x = 4.0
        self.target_y = 0.0
        self.target_threshold = 0.5  # Distance threshold for reaching target
        self.terminal_bonus = 1000.0 # Reward for reaching target
        
        
        # Filter controllable joints (ignore fixed joints)
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robotId, i)
            joint_name = joint_info[1].decode('utf-8') # Get joint name

            if joint_info[2] != p.JOINT_FIXED:  # Non-fixed joints
                self.joint_indices.append(i)
                self.joint_names.append(joint_name)

                if "leg" in joint_name:
                    self.foot_joint_indices.append(i)
        

        # Print identified joint information
        print(f"找到 {len(self.joint_indices)} 个可控关节: {self.joint_names}")
        print(f"脚关节索引: {self.foot_joint_indices}")

        # Define action and observation space dimensions
        self.action_dim = len(self.joint_indices)

        '''
        Observation space dimensions:
        - Torso position (y, z): 2
        - Torso velocity (x, y, z): 3
        - Torso rotation angles (roll, pitch, yaw): 3
        - Torso angular velocity: 3
        - Joint positions and velocities: 8
        - Foot-ground contact forces: 4
        - Lowest leg height: 1
        - Previous action: 4 (torque for 4 joints)
        - Torso position x: 1
        '''
        self.obs_dim = 2 + 3 + 6 + 8 + 4 + 4 + 1 + 1 # Custom observation dimension

        # Initialize environment state
        self.reset()

        # Store previous step data for reward calculation
        self.prev_torque = np.zeros(self.action_dim)
        self.prev_action = np.zeros(self.action_dim)

        self.current_steps = 0  # Step counter

    def reset(self):
        # Reset robot position and orientation
        p.resetBasePositionAndOrientation(self.robotId, [0, 0, 0.20], [0, 0, 0, 1])

        self.stagnation_counter = 0 # Reset stagnation counter

        self.current_steps = 0
        self.goal_speed = 0.3
        
        # Reset joint positions to initial state
        for i in self.joint_indices:
            p.resetJointState(self.robotId, i, targetValue=0, targetVelocity=0)

            p.setJointMotorControl2(
            bodyIndex=self.robotId,
            jointIndex=i,
            controlMode=p.VELOCITY_CONTROL,
            force=0  # Disable default motor
            )
        
        # Get initial observation
        self.last_action = np.zeros(self.action_dim)   # Initialize previous action cache
        observation = self._get_observation()

        self.timestep = 0  # Initialize timestep
        
        return observation
    
    def step(self, action):

        self.current_steps += 1

        # Apply joint torques
        max_torque = 8.0
        for _ in range(self.action_repeat):
            for i, joint_idx in enumerate(self.joint_indices):
                torque = float(action[i]) * max_torque 
                p.setJointMotorControl2(
                    bodyIndex=self.robotId,
                    jointIndex=joint_idx,
                    controlMode=p.TORQUE_CONTROL,
                    force=torque
                )
            p.stepSimulation()

        observation = self._get_observation()

        base_reward = self._calculate_reward(observation, action)

        # Calculate terminal reward
        done = self._check_done(observation)

        terminal_bonus = 0.0

        if done:
            # Calculate current 2D distance to target
            pos_x = observation[-1]
            pos_y = observation[0]
            dx = pos_x - self.target_x
            dy = pos_y - self.target_y
            dist2d = np.sqrt(dx*dx + dy*dy)
            # Trigger terminal reward if within threshold
            if dist2d < self.target_threshold:
                terminal_bonus = self.terminal_bonus
        
        reward = base_reward + terminal_bonus

        # Save current action and torque for next calculation
        self.prev_action = action
        self.prev_torque = self._get_joint_torques()

        done = self._check_done(observation)
        
        # Update previous action cache
        self.last_action = action
        
        info = {}  # Optional additional information
        
        return observation, reward, done, info
    

    def _get_joint_torques(self):
        # Get current joint torques
        torques = []
        for i in self.joint_indices:
            _, _, _, applied_torque = p.getJointState(self.robotId, i)
            torques.append(applied_torque)
        return np.array(torques)

    def _get_foot_contacts(self):
        # Get foot contact forces and lowest point
        foot_contacts = [0.0] * len(self.foot_joint_indices)
        min_foot_height = 99
        contact_points = p.getContactPoints(bodyA=self.robotId, bodyB=self.planeId)
        for contact in contact_points:
            contact_joint_idx = contact[3]
            contact_pos_z = contact[6][2]

            if contact_pos_z < min_foot_height:
                min_foot_height = contact_pos_z
            # Check if this link is in the leg links list

            if contact_joint_idx in self.foot_joint_indices:
                idx = self.foot_joint_indices.index(contact_joint_idx)
                foot_contacts[idx] += contact[9] 
                break
            
        if min_foot_height > 90:
            min_foot_height = 0.0

        return foot_contacts,min_foot_height

    def _get_observation(self):
        # Get torso position (y and z directions)
        position, _ = p.getBasePositionAndOrientation(self.robotId)
        pos_x, pos_y, pos_z = position
        # print(f"pos_y: {pos_y}, pos_z: {pos_z}")
        
        # Get torso velocity (x,y,z) and angular velocity
        linear_velocity, angular_velocity = p.getBaseVelocity(self.robotId)
        vel_x, vel_y, vel_z = linear_velocity
        ang_vel_x, ang_vel_y, ang_vel_z = angular_velocity
        # print(f"vel_x: {vel_x}, vel_y: {vel_y}, vel_z: {vel_z}")
        
        # Get torso rotation angles (quaternion to euler)
        _, orientation = p.getBasePositionAndOrientation(self.robotId)
        euler_angles = p.getEulerFromQuaternion(orientation)
        roll, pitch, yaw = euler_angles
        
        # Combine rotation angles and angular velocity
        rotation_state = [roll, pitch, yaw, ang_vel_x, ang_vel_y, ang_vel_z]
    
        # Get joint positions and velocities
        joint_states = []
        for i in self.joint_indices:
            joint_pos, joint_vel, _, _ = p.getJointState(self.robotId, i)
            joint_states.extend([joint_pos, joint_vel])
            # print(f"Joint {i} pos: {joint_pos}, vel: {joint_vel}")
        
        # Get foot-ground contact forces (4 legs)
        foot_contacts, min_height = self._get_foot_contacts()

        # Combine all observations (note dimension changes)
        observation = [
            pos_y, pos_z,          
            vel_x, vel_y, vel_z,  
            *rotation_state,      
            *joint_states,       
            *foot_contacts,       
            min_height,         
            *self.last_action       
        ]
        observation.append(pos_x)  # Add torso x position
        
        return np.array(observation, dtype=np.float32)
    
    def set_reward_weights(self, energy_weight, path_weight):
        ## Set penalty weights in reward function
        self.energy_weight = energy_weight
        self.path_weight = path_weight

    def _calculate_reward(self, observation, action):#TODO
        # Reward function design - needs adjustment based on task objectives
        pos_x = observation[-1]     # Torso x position (should stay near target)
        pos_y = observation[0]      # Torso y position (should stay near 0)
        pos_z = observation[1]      # Torso z position (should maintain proper height)
        vel_x = observation[2]      # Torso x velocity
        vel_y = observation[3]      # Torso y velocity
        vel_z = observation[4]      # Torso z velocity
        roll = observation[5]       # Torso roll angle
        pitch = observation[6]      # Torso pitch angle
        yaw = observation[7]        # Torso yaw angle

        upright_target_z = 0.2      # Ideal height
        fall_penalty = -5.0 if pos_z < 0.15 else 0.0

        # —— 1. Torso height reward —— #
        K1 = 15.0
        upright_reward = -K1 * (pos_z - upright_target_z) ** 2

        # —— 2. Torso pitch yaw roll quadratic reward —— #
        K2 = 1.0
        balance_reward = np.exp(-K2 * ((abs(pitch) + abs(roll) + abs(yaw)) ** 2))

        # —— 3. Body velocity penalty —— #
        forward_reward = vel_x * 2.0

        # —— 4. Energy consumption penalty —— #
        K_e = 0.1
        energy_penalty = -K_e * np.sum( action ** 2)

        # —— 5. Centerline deviation penalty —— #
        K_p = 3.0
        path_penalty = np.exp(-K_p * (pos_y ** 2))

        # —— 6. Target position reward —— #
        dx = pos_x - self.target_x
        dy = pos_y - self.target_y
        dist2d = np.sqrt(dx * dx + dy * dy)
        k_dist = 0.3  # Decay rate
        distance_reward = 8.0 * np.exp(-k_dist * dist2d ** 2)

        reward = (
            forward_reward
            + upright_reward
            + balance_reward
            + distance_reward
            + 0.0625
            + energy_penalty
            # + path_penalty
        )

        return reward
    
    def _check_done(self, observation):
        pos_x = observation[-1]     # Torso x position
        pos_y = observation[0]      # Torso y position
        pos_z = observation[1]      # Torso z position
        roll = observation[5]       # Torso roll angle
        pitch = observation[6]      # Torso pitch angle
        yaw = observation[7]        # Torso yaw angle
        vel_x = observation[2]      # Torso x velocity

        # Maximum steps reached #
        maximum_step = self.current_steps >= self.max_steps

        # Close enough to target #
        dx = pos_x - self.target_x
        dy = pos_y - self.target_y
        dist2d = np.sqrt(dx * dx + dy * dy)
        reach_target = dist2d < self.target_threshold

        # Check if fallen or stagnant
        fallen = (
            pos_z < 0.10
        )

        # Check body orientation
        direction = (
            roll >1.07 or   # ~60 degrees
            pitch > 1.57 or # ~90 degrees
            yaw > 1.07      # ~60 degrees 
        )

        if vel_x < 0.05:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        stagnant = self.stagnation_counter > 50  # ~0.2 seconds of stagnation

        if fallen:
            print("Robot has fallen!")
        elif stagnant:
            print("Robot is stagnant!")
        elif direction:
            print("Robot is not upright!")
        elif maximum_step:
            print("Reached maximum step count, ending episode.")
        elif reach_target:
            print(f"Reached target within {self.target_threshold}m, ending episode.")

        return fallen or stagnant or direction or maximum_step or reach_target
    
    def close(self):
        p.disconnect()

