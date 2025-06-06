import pybullet as p
import pybullet_data
import numpy as np
import os

class RobotEnv:
    def __init__(self, render=False):
        # 初始化PyBullet环境
        if render:
            self.physicsClient = p.connect(p.GUI)  # 图形界面模式
        else:
            self.physicsClient = p.connect(p.DIRECT)  # 无渲染模式
        
        # 设置物理模拟参数
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 加载PyBullet自带的模型
        p.setGravity(0, 0, -9.8)  # 设置重力
        p.setTimeStep(1.0/240.0)  # 设置时间步长
        
        # 加载地面
        self.planeId = p.loadURDF("plane.urdf")
        
        # 加载机器人模型
        robot_urdf_path = "dog.urdf"  
        self.robotId = p.loadURDF(robot_urdf_path, [0, 0, 0.60]) # 机器人初始位置
        
        # 获取关节信息
        self.num_joints = p.getNumJoints(self.robotId)
        self.joint_indices = []
        self.joint_names = []
        self.foot_joint_indices = [] 
        self.action_repeat = 2  # 每个动作重复执行的次数
        self.stagnation_counter = 0  # 停滞计数器

        # 控制信息
        self.max_steps = 2400
        self.target_x = 4.0
        self.target_y = 0.0
        self.target_threshold = 0.5  # 到达目标的距离阈值
        self.terminal_bonus = 1000.0 # 到达目标的奖励
        
        
        # 过滤出可控关节（忽略固定关节）
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robotId, i)
            joint_name = joint_info[1].decode('utf-8') # 获取关节名称

            if joint_info[2] != p.JOINT_FIXED:  # 非固定关节
                self.joint_indices.append(i)
                self.joint_names.append(joint_name)

                if "leg" in joint_name:
                    self.foot_joint_indices.append(i)
        

        # 打印识别出的关节信息
        print(f"找到 {len(self.joint_indices)} 个可控关节: {self.joint_names}")
        print(f"脚关节索引: {self.foot_joint_indices}")

        # 定义动作和观测空间维度
        self.action_dim = len(self.joint_indices)  # 关节数量，对应六个扭矩控制

        '''
        观测空间维度：
        - 躯干位置 (y, z): 2
        - 躯干速度 (x, y, z): 3
        - 躯干旋转角度 (roll, pitch, yaw): 3
        - 躯干角速度 (angular velocity): 3
        - 关节位置和速度: 8
        - 脚与地面的接触力: 4
        - 最低腿高度: 1
        - 上一次动作: 4 (对应4个关节的扭矩)
        - 躯干位置 x: 1
        '''
        self.obs_dim = 2 + 3 + 6 + 8 + 4 + 4 + 1 + 1# 根据需求定义的观测维度

        # 初始化环境状态
        self.reset()

        # 存储前一步的数据用于奖励计算
        self.prev_torque = np.zeros(self.action_dim)
        self.prev_action = np.zeros(self.action_dim)

        self.current_steps = 0  # 新增步数计数器

    def reset(self):
        # 重置机器人位置和姿态
        p.resetBasePositionAndOrientation(self.robotId, [0, 0, 0.20], [0, 0, 0, 1])

        self.stagnation_counter = 0  # 重置停滞计数器

        self.current_steps = 0
        self.goal_speed = 0.3
        
        # 重置关节位置到初始状态
        for i in self.joint_indices:
            p.resetJointState(self.robotId, i, targetValue=0, targetVelocity=0)

            p.setJointMotorControl2(
            bodyIndex=self.robotId,
            jointIndex=i,
            controlMode=p.VELOCITY_CONTROL,
            force=0  # 禁用默认电机
            )
        
        # 获取初始观测
        self.last_action = np.zeros(self.action_dim)  # 初始化上一次动作缓存
        observation = self._get_observation()

        self.timestep = 0  # 初始化时间步
        
        return observation
    
    def step(self, action):

        self.current_steps += 1

        # 应用关节扭矩
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

        # 计算终局奖励
        done = self._check_done(observation)

        terminal_bonus = 0.0

        if done:
            # 计算当前距离目标的 2D 平面距离
            pos_x = observation[-1]
            pos_y = observation[0]
            dx = pos_x - self.target_x
            dy = pos_y - self.target_y
            dist2d = np.sqrt(dx*dx + dy*dy)
            # 只要 dist2d < threshold 就算“到达触发终局奖励”
            if dist2d < self.target_threshold:
                terminal_bonus = self.terminal_bonus
        
        reward = base_reward + terminal_bonus

        # 保存当前动作和扭矩用于下次计算
        self.prev_action = action
        self.prev_torque = self._get_joint_torques()

        done = self._check_done(observation)
        
        # 更新上一次动作缓存
        self.last_action = action
        
        info = {}  # 可选的额外信息
        
        return observation, reward, done, info
    

    def _get_joint_torques(self):
        """获取当前关节扭矩"""
        torques = []
        for i in self.joint_indices:
            _, _, _, applied_torque = p.getJointState(self.robotId, i)
            torques.append(applied_torque)
        return np.array(torques)

    def _get_foot_contacts(self):
        """获取足部接触力与最低点"""
        foot_contacts = [0.0] * len(self.foot_joint_indices)
        min_foot_height = 99
        contact_points = p.getContactPoints(bodyA=self.robotId, bodyB=self.planeId)
        for contact in contact_points:
            contact_joint_idx = contact[3]
            contact_pos_z = contact[6][2]

            if contact_pos_z < min_foot_height:
                min_foot_height = contact_pos_z
             # 检查这个链接是否在腿链接列表中

            if contact_joint_idx in self.foot_joint_indices:
                idx = self.foot_joint_indices.index(contact_joint_idx)
                foot_contacts[idx] += contact[9]  # 法向力
                break
            
        if min_foot_height > 90:
            min_foot_height = 0.0

        return foot_contacts,min_foot_height

    def _get_observation(self):
        # 获取躯干位置（y和z方向）
        position, _ = p.getBasePositionAndOrientation(self.robotId)
        pos_x, pos_y, pos_z = position
        # print(f"pos_y: {pos_y}, pos_z: {pos_z}")
        
        # 获取躯干速度（x,y,z）和角速度
        linear_velocity, angular_velocity = p.getBaseVelocity(self.robotId)
        vel_x, vel_y, vel_z = linear_velocity
        ang_vel_x, ang_vel_y, ang_vel_z = angular_velocity
        # print(f"vel_x: {vel_x}, vel_y: {vel_y}, vel_z: {vel_z}")
        
        # 获取躯干旋转角度（四元数转欧拉角）
        _, orientation = p.getBasePositionAndOrientation(self.robotId)
        euler_angles = p.getEulerFromQuaternion(orientation)
        roll, pitch, yaw = euler_angles
        
        # 组合旋转角度和角速度
        rotation_state = [roll, pitch, yaw, ang_vel_x, ang_vel_y, ang_vel_z]
    
        # 获取关节位置和速度
        joint_states = []
        for i in self.joint_indices:
            joint_pos, joint_vel, _, _ = p.getJointState(self.robotId, i)
            joint_states.extend([joint_pos, joint_vel])
            # print(f"Joint {i} pos: {joint_pos}, vel: {joint_vel}")
        
        # 获取腿与地面的接触力（4条腿）
        foot_contacts, min_height = self._get_foot_contacts()

        # 组合所有观测（注意维度变化）
        observation = [
            pos_y, pos_z,          # 2
            vel_x, vel_y, vel_z,   # 3
            *rotation_state,       # 6（roll, pitch, yaw, ang_vel_x, ang_vel_y, ang_vel_z）
            *joint_states,         # 8（4个关节的位置和速度）
            *foot_contacts,        # 4（每条腿的接触力）
            min_height,            # 1（最低腿高度）
            *self.last_action       # 4（上一次动作）
        ]
        observation.append(pos_x)  # 添加躯干x位置
        
        return np.array(observation, dtype=np.float32)
    
    def set_reward_weights(self, energy_weight, path_weight):
        # 设置奖励函数中的惩罚权重
        self.energy_weight = energy_weight
        self.path_weight = path_weight

    def _calculate_reward(self, observation, action):#TODO
        # 奖励函数设计，这里需要根据任务目标进行调整
        pos_x = observation[-1]  # 躯干x方向位置（希望保持在目标位置附近）
        pos_y = observation[0]  # 躯干y方向位置（希望保持在0附近）
        pos_z = observation[1]  # 躯干z方向位置（希望保持在合适高度）
        vel_x = observation[2]  # 躯干x方向速度
        vel_y = observation[3]  # 躯干y方向速度
        vel_z = observation[4]  # 躯干z方向速度
        roll = observation[5]  # 躯干roll角度
        pitch = observation[6]  # 躯干pitch角度
        yaw = observation[7] # 躯干yaw角度
        min_height = observation[-5]  # 最低腿高度
        foot_contact = observation[-9:-5]  # 脚与地面的接触力

        upright_target_z = 0.2       # 理想高度
        fall_penalty = -5.0 if pos_z < 0.15 else 0.0

        # —— 1. 躯干高度奖励 —— #
        K1 = 15.0
        # upright_reward = np.exp(-K1 * (pos_z - upright_target_z) ** 2)
        upright_reward = -K1 * (pos_z - upright_target_z)**2
        # upright_reward = 5 * ((pos_z-min_height - 0.56) ** 2)

        # —— 2. 躯干 pitch yaw roll “二次”奖励 —— #
        K2 = 1.0
        balance_reward = np.exp(-K2 * ((abs(pitch) + abs(roll) + abs(yaw)) ** 2))

        # —— 3. 身体速度惩罚 —— #
        #vel_penalty = -K_vel * (vel_x ** 2 + vel_y ** 2 + vel_z ** 2)
        # forward_reward = np.exp(-abs(vel_x - self.goal_speed))
        # forward_reward = -abs(vel_x - self.goal_speed)
        # forward_reward = np.exp(- (vel_x - 0.3)**2 )
        forward_reward = vel_x * 2.0

        # —— 4. 能量消耗惩罚 —— #
        K_e = 0.1
        energy_penalty = -K_e * np.sum( action ** 2)
        # energy_penalty = np.exp(-K_e * np.sum(action**2))

        # —— 5. 偏离中心线惩罚 —— #
        K_p = 3.0
        # path_penalty = -K_p * (pos_y ** 2)
        path_penalty = np.exp(-K_p * (pos_y ** 2))

        # —— 6. 速度过慢惩罚 —— #
        stagnation_penalty = -1.0 if abs(vel_x) < 0.05 else 0.0

        # —— 7. 目标位置奖励 —— #
        dx = pos_x - self.target_x
        dy = pos_y - self.target_y
        dist2d = np.sqrt(dx * dx + dy * dy)
        k_dist = 0.3  # 衰减速率，可根据需求调整，例如 0.2、1.0 等
        distance_reward = 8.0 * np.exp(-k_dist * dist2d ** 2)

        reward = (
            forward_reward
            + upright_reward
            + balance_reward
            + distance_reward
            + 0.0625
            + energy_penalty
            # + path_penalty
        )# #TODO: 调整各项奖励的权重

        return reward
    
    def _check_done(self, observation):
        pos_x = observation[-1]   # 躯干 x 方向位置
        pos_y = observation[0]    # 躯干 y 方向位置
        pos_z = observation[1]
        roll = observation[5]  # 躯干roll角度
        pitch = observation[6]  # 躯干pitch角度
        yaw = observation[7] # 躯干yaw角度
        vel_x = observation[2]  # 躯干x方向速度

        # —— 1. 如果步数已经到达最大值，则结束 —— #
        maximum_step = self.current_steps >= self.max_steps

        # —— 2. 如果已经足够接近目标点，也结束 —— #
        dx = pos_x - self.target_x
        dy = pos_y - self.target_y
        dist2d = np.sqrt(dx * dx + dy * dy)
        reach_target = dist2d < self.target_threshold

        # 检查是否摔倒或停滞
        fallen = (
            pos_z < 0.10
        )

        # 检查躯体方向是否正确
        direction = (
            roll >1.07 or # 约60度
            pitch > 1.57 or # 约90度
            yaw > 1.07 # 约60度 
        )

        if vel_x < 0.05:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        stagnant = self.stagnation_counter > 50  # 约0.2秒停滞

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
        # return False
    #TODO
    
    def close(self):
        p.disconnect()

