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
        robot_urdf_path = "simple_robot.urdf"  
        self.robotId = p.loadURDF(robot_urdf_path, [0, 0, 0.60]) # 机器人初始位置
        
        # 获取关节信息
        self.num_joints = p.getNumJoints(self.robotId)
        self.joint_indices = []
        self.joint_names = []
        self.foot_joint_indices = []  # 存储脚踝关节索引
        self.action_repeat = 2  # 每个动作重复执行的次数
        
        
        # 过滤出可控关节（忽略固定关节）
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robotId, i)
            joint_name = joint_info[1].decode('utf-8') # 获取关节名称

            if joint_info[2] != p.JOINT_FIXED:  # 非固定关节
                self.joint_indices.append(i)
                self.joint_names.append(joint_name)

                # 检查是否为脚踝关节
                if "ankle" in joint_name:
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
        - 关节位置和速度: 12 
        - 脚与地面的接触力: 2
        - 上一次动作: 2 (对应6个关节的扭矩)
        '''
        self.obs_dim = 2 + 3 + 6 + 12 + 2 + self.action_dim  # 根据需求定义的观测维度

        self.phase = 2

        # 初始化环境状态
        self.reset()

        # 添加步态时钟相关参数
        self.gait_cycle_time = 2.0  # 步态周期时长(秒)
        self.phase_clock = 0.0  # 步态相位 [0,1)
        self.time_in_cycle = 0.0  # 当前周期内的时间

        # 步态时钟函数
        self.right_clock = self._create_phase_functions("right")
        self.left_clock = self._create_phase_functions("left")

        # 存储前一步的数据用于奖励计算
        self.prev_torque = np.zeros(self.action_dim)
        self.prev_action = np.zeros(self.action_dim)

        self.current_steps = 0  # 新增步数计数器

    def set_phase(self, phase):
        """phase 必须是 1、2 或 3。训练循环中根据 Episode 调用本函数。"""
        assert phase in (1, 2, 3), "phase 必须为 1、2 或 3"
        self.phase = phase

    def _create_phase_functions(self, side):
        """创建足部力和速度的时钟函数"""
        # 左腿先摆动
        if side == "right":
            def frc_fn(phase_clock):
                return 1.0 if phase_clock < 0.5 else -1.0  # 右腿在前半周期支撑

            def vel_fn(phase_clock):
                return -1.0 if phase_clock < 0.5 else 1.0  # 右腿在后半周期摆动
        else:
            def frc_fn(phase_clock):
                return -1.0 if phase_clock < 0.5 else 1.0  # 左腿在后半周期支撑

            def vel_fn(phase_clock):
                return 1.0 if phase_clock < 0.5 else -1.0  # 左腿在前半周期摆动

        return [frc_fn, vel_fn]

    def _calc_foot_frc_clock_reward(self):
        """基于步态时钟的足部力奖励"""
        r_frc = self.right_clock[0](self.phase_clock)
        l_frc = self.left_clock[0](self.phase_clock)

        # 获取足部接触力
        foot_contacts,min_height = self._get_foot_contacts()
        l_contact = foot_contacts[0] if len(foot_contacts) > 0 else 0
        r_contact = foot_contacts[1] if len(foot_contacts) > 1 else 0

        # 归一化
        max_force = 100  # 假设最大接触力
        norm_r = min(r_contact, max_force) / max_force
        norm_l = min(l_contact, max_force) / max_force

        # 计算匹配得分
        r_score = np.tan(np.pi / 4 * r_frc * (2 * norm_r - 1))
        l_score = np.tan(np.pi / 4 * l_frc * (2 * norm_l - 1))

        return (r_score + l_score) / 2,min_height

    def _calc_foot_vel_clock_reward(self):
        """基于步态时钟的足部速度奖励"""
        r_vel = self.right_clock[1](self.phase_clock)
        l_vel = self.left_clock[1](self.phase_clock)

        # 获取足部速度
        r_foot_vel = self._get_foot_velocity("right")
        l_foot_vel = self._get_foot_velocity("left")

        # 归一化
        max_vel = 2.0  # 假设最大速度
        norm_r = min(np.linalg.norm(r_foot_vel), max_vel) / max_vel
        norm_l = min(np.linalg.norm(l_foot_vel), max_vel) / max_vel

        # 计算匹配得分
        r_score = np.tan(np.pi / 4 * r_vel * (2 * norm_r - 1))
        l_score = np.tan(np.pi / 4 * l_vel * (2 * norm_l - 1))

        return (r_score + l_score) / 2

    def _get_foot_velocity(self, side):
        """获取足部速度"""
        foot_idx = self.foot_joint_indices[0] if side == "left" else self.foot_joint_indices[1]
        link_state = p.getLinkState(self.robotId, foot_idx, computeLinkVelocity=1)
        return link_state[6]  # 返回线速度

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
            for i, foot_idx in enumerate(self.foot_joint_indices):
                if contact_joint_idx == foot_idx:
                    foot_contacts[i]+= contact[9]  # 法向力
                    break
        if min_foot_height>90:
            min_foot_height = 0.0
        return foot_contacts,min_foot_height

    def reset(self):
        # 重置机器人位置和姿态
        p.resetBasePositionAndOrientation(self.robotId, [0, 0, 0.60], [0, 0, 0, 1])

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
        # 更新步态时钟
        self.current_steps += 1
        self.time_in_cycle += 1.0/240.0
        self.phase_clock = (self.time_in_cycle % self.gait_cycle_time) / self.gait_cycle_time

        # 应用关节扭矩
        max_torque = 10.0
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

        reward = self._calculate_reward(observation, action)

        # 保存当前动作和扭矩用于下次计算
        self.prev_action = action
        self.prev_torque = self._get_joint_torques()

        done = self._check_done(observation)
        
        # 更新上一次动作缓存
        self.last_action = action
        
        info = {}  # 可选的额外信息
        
        return observation, reward, done, info
    
    def _get_observation(self):
        # 获取躯干位置（y和z方向）
        position, _ = p.getBasePositionAndOrientation(self.robotId)
        pos_y, pos_z = position[1], position[2]
        # print(f"pos_y: {pos_y}, pos_z: {pos_z}")
        
        # 获取躯干速度（x,y,z）和角速度
        linear_velocity, angular_velocity = p.getBaseVelocity(self.robotId)
        vel_x, vel_y, vel_z = linear_velocity
        ang_vel_x, ang_vel_y, ang_vel_z = angular_velocity
        # print(f"vel_x: {vel_x}, vel_y: {vel_y}, vel_z: {vel_z}")
        
        # 获取躯干旋转角度
        # 注意：PyBullet返回四元数，需要转换为欧拉角
        _, orientation = p.getBasePositionAndOrientation(self.robotId)
        euler_angles = p.getEulerFromQuaternion(orientation)
        roll, pitch, yaw = euler_angles
        # print(f"roll: {roll}, pitch: {pitch}, yaw: {yaw}")
        
        # 组合旋转角度和角速度
        rotation_state = [roll, pitch, yaw] + list(angular_velocity)
        
        # 获取关节位置和速度
        joint_states = []
        for i in self.joint_indices:
            joint_pos, joint_vel, _, _ = p.getJointState(self.robotId, i)
            joint_states.extend([joint_pos, joint_vel])
            # print(f"Joint {i} pos: {joint_pos}, vel: {joint_vel}")
        
        # 获取脚与地面的接触力
        foot_contacts = [0.0] * len(self.foot_joint_indices)
        contact_points = p.getContactPoints(bodyA=self.robotId, bodyB=self.planeId)
        for contact in contact_points:
            contact_joint_idx  = contact[3]  # 接触的关节索引
            for i, foot_idx in enumerate(self.foot_joint_indices):
                if contact_joint_idx == foot_idx:
                    foot_contacts[i] = contact[9]  # 法向力
                    # print(f"Foot {i} contact force: {foot_contacts[i]}")
                    break
        
        # 组合所有观测
        observation = [pos_y, pos_z] + [vel_x, vel_y, vel_z] + rotation_state + joint_states + foot_contacts + list(self.last_action)
        
        return np.array(observation, dtype=np.float32)
    
    def set_reward_weights(self, energy_weight, path_weight):
        # 设置奖励函数中的惩罚权重
        self.energy_weight = energy_weight
        self.path_weight = path_weight

    def _calculate_reward(self, observation, action):#TODO
        # 奖励函数设计，这里需要根据任务目标进行调整
        pos_y = observation[0]  # 躯干y方向位置（希望保持在0附近）
        pos_z = observation[1]  # 躯干z方向位置（希望保持在合适高度）
        vel_x = observation[2]  # 躯干x方向速度
        vel_y = observation[3]  # 躯干y方向速度
        vel_z = observation[4]  # 躯干z方向速度
        roll = observation[5]  # 躯干roll角度
        pitch = observation[6]  # 躯干pitch角度
        yaw = observation[7] # 躯干yaw角度

        ankle_angle_vel = [observation[16], observation[22]]  # 脚踝角速度
        foot_contacts = [observation[-8], observation[-7]]  # 获取脚与地面的接触力

        # 新增步态时钟奖励
        gait_frc_reward,min_height= self._calc_foot_frc_clock_reward()
        gait_vel_reward = self._calc_foot_vel_clock_reward()


        upright_target_z = 0.56       # 理想高度
        fall_penalty = -5.0 if pos_z < 0.3 else 0.0
        self.goal_speed = 0.4

        if self.phase == 1:
            # ===== 阶段 1：只做“保持直立”+“保持平衡”奖励 =====

            # —— 1. 躯干高度奖励 —— #
            K1 = 50.0
            upright_reward = np.exp(-K1 * (pos_z - upright_target_z) ** 2)

            # —— 2. 躯干 pitch “二次”奖励 —— #
            K2 = 10.0
            balance_reward = np.exp(-K2 * (pitch ** 2))

            # —— 3. 身体速度惩罚 —— #
            K_vel = 5.0
            vel_penalty = -K_vel * (vel_x**2 + vel_y**2 + vel_z**2)
            # vel_penalty = np.exp(-K_vel * (vel_x**2 + vel_y**2 + vel_z**2))

            # —— 4. 能量消耗惩罚 —— #
            K_e = 2.0
            energy_penalty = -K_e * np.sum(action**2)
            # energy_penalty = np.exp(-K_e * np.sum(action**2))

            # —— 5. 偏离中心线惩罚 —— #
            K_p = 3.0
            path_penalty = -K_p * (pos_y ** 2)
            # path_penalty = np.exp(-K_p * (pos_y ** 2))

            reward = (upright_reward + balance_reward 
                      + vel_penalty + fall_penalty 
                      + path_penalty + energy_penalty)

        elif self.phase == 2: #TODO
            # —— 1. 躯干高度奖励 —— #
            K1 = 50.0
            upright_reward = np.exp(-K1 * (pos_z-min_height - 0.56) ** 2)

            # —— 2. 躯干 pitch yaw roll “二次”奖励 —— #
            K2 = 10.0
            balance_reward = np.exp(-K2 * ((pitch + yaw + roll) ** 2))

            # —— 3. 身体速度惩罚 —— #
            #vel_penalty = -K_vel * (vel_x ** 2 + vel_y ** 2 + vel_z ** 2)
            forward_reward=np.exp(-abs(vel_x - self.goal_speed))

            # —— 4. 能量消耗惩罚 —— #
            K_e = 0.25
            energy_penalty = np.exp(-K_e * np.sum(action**2))

            # —— 5. 偏离中心线惩罚 —— #
            K_p = 3.0
            #path_penalty = -K_p * (pos_y ** 2)
            path_penalty = np.exp(-K_p * (pos_y ** 2))

            time_reward=-np.exp(-min(self.current_steps*0.01,10) if abs(pitch) < 0.5 else 0)

            #TODO: 调整各项奖励的权重
            reward = (4.0 * forward_reward + 
                      1.0 * upright_reward + 
                      2.0 * balance_reward + 
                      1.0 * gait_frc_reward + 
                      1.0 * gait_vel_reward + 
                      1.0 * path_penalty + 
                      2.0 * energy_penalty + 
                      4.0 * time_reward)
            
        #print(forward_reward,upright_reward,balance_reward,gait_frc_reward,gait_vel_reward,path_penalty,energy_penalty,time_reward)
        
        else:  # self.phase == 3
            reward = 0.0
        
        # 映射奖励到[-1, 1]范围 TODO
        # reward = np.tanh(reward)

        return reward
    
    def _check_done(self, observation):
        pos_z = observation[1]  # 躯干z方向位置
        roll = observation[5]
        pitch = observation[6]
        yaw = observation[7]
        return pos_z < 0.05 or abs(pitch) > 1.37 or abs(roll) > 1.37 or abs(yaw) > 1.37
    #TODO
    
    def close(self):
        p.disconnect()