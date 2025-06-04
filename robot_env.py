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
        - 上一次动作: 6 (对应6个关节的扭矩)
        '''
        self.obs_dim = 2 + 3 + 6 + 12 + 2 + self.action_dim  # 根据需求定义的观测维度

        # ====== 新增：阶段控制变量，默认从阶段 1 开始 ======
        #  phase = 1, 2 或 3，由训练循环根据 Episode 设置
        '''
        阶段 1(Episode 0 - 499): 只给“保持直立”+“保持躯干水平”奖励，让机器人先学会不倒；

        阶段 2(Episode 500 - 999): 在阶段 1 的基础上，加入少量向前移动奖励，让它开始尝试往前挪动；

        阶段 3(Episode ≥ 1000): 加大向前移动奖励，并加上能耗惩罚，鼓励高效步态。
        '''
        self.phase = 1

        
        # 初始化环境状态
        self.reset()
    
    def set_phase(self, phase):
        """phase 必须是 1、2 或 3。训练循环中根据 Episode 调用本函数。"""
        assert phase in (1, 2, 3), "phase 必须为 1、2 或 3"
        self.phase = phase

    def reset(self):
        # 重置机器人位置和姿态
        p.resetBasePositionAndOrientation(self.robotId, [0, 0, 0.60], [0, 0, 0, 1])
        
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
        
        return observation
    
    def step(self, action):
        # 应用关节扭矩
        max_torque = 1.5
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
        # 执行一步物理模拟
        # print("Before stepSimulation")
        # p.stepSimulation()
        # print("After stepSimulation")
        
        # 获取新的观测
        observation = self._get_observation()
        
        # 计算奖励
        reward = self._calculate_reward(observation, action)
        
        # 检查是否终止
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

    def _calculate_reward(self, observation, action):
        # 奖励函数设计，这里需要根据任务目标进行调整
        foot_contacts = [0.0] * 2
        pos_y = observation[0]  # 躯干y方向位置（希望保持在0附近）
        pos_z = observation[1]  # 躯干z方向位置（希望保持在合适高度）
        vel_x = observation[2]  # 躯干x方向速度
        vel_y = observation[3]  # 躯干y方向速度
        vel_z = observation[4]  # 躯干z方向速度
        pitch = observation[6]  # 躯干pitch角度（希望接近0，表示躯干竖直）
        foot_contacts[0] = observation[-8]
        foot_contacts[1] = observation[-7]
        
        upright_target_z = 0.56       # 理想高度
        fall_penalty = -5.0 if pos_z < 0.3 else 0.0

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

        elif self.phase == 2:#TODO
            # ===== Phase 2：保持姿态 & 双脚着地 & 能微移向前 =====
            base_reward = 0.0  # 基础奖励

            # —— 1. 躯干高度奖励 —— #
            K1 = 50.0
            # upright_reward = -K1 * (pos_z - upright_target_z) ** 2
            upright_reward = np.exp(-K1 * (pos_z - upright_target_z) ** 2)

            # 额外惩罚过低的高度（防止蹲下）
            if pos_z < 0.4:
                upright_reward *= 0.5  # 降低奖励
                height_penalty = -20.0 * (0.4 - pos_z)  # 线性惩罚过低高度
            else:
                height_penalty = 0.0

            # —— 2. 躯干 pitch 惩罚 —— #
            K2 = 100.0
            # # balance_reward = np.exp(-K2 * (pitch ** 2))
            balance_penalty = -K2 * (pitch ** 2)

            # —— 4. 能量消耗惩罚 —— #
            K_e = 0.1
            # energy_penalty = -K_e * np.sum(action**2)
            energy_penalty = -K_e * np.sum(action**2)

            # —— 5. 偏离中心线惩罚 —— #
            K_p = 3.0
            # path_penalty = -K_p * (pos_y ** 2)
            path_penalty = -K_p * (pos_y ** 2)

            # —— 6. 向前移动奖励 —— #
            K_forward2 = 5.0
            forward_reward = K_forward2 * max(vel_x, 0)

            # —— 7. 抬脚奖励 —— #
            K_lift_foot = 2.0
            lift_foot_reward = 0
            
            # 区分左右脚，鼓励交替抬脚
            left_foot_force = foot_contacts[0]
            right_foot_force = foot_contacts[1]
            
            # 计算抬脚的平衡度（理想情况：一只脚完全抬起，另一只脚完全着地）
            lift_balance = abs(left_foot_force - right_foot_force) / 5.0  # 归一化
            lift_foot_reward = lift_balance * K_lift_foot

            # 额外奖励合理的抬脚高度（通过关节角度判断）
            foot_lift_reward = 0
            if left_foot_force < 0.5:  # 左脚抬起
                foot_lift_reward += 1.0
            if right_foot_force < 0.5:  # 右脚抬起
                foot_lift_reward += 1.0

            fall_penalty = -50.0 if pos_z < 0.3 else 0.0

            reward = forward_reward  + upright_reward + energy_penalty + path_penalty + balance_penalty + fall_penalty + lift_foot_reward \
                        + foot_lift_reward + height_penalty
        
        else:  # self.phase == 3
            # ===== Phase 3：能微移向前 =====
            
            # —— 1. 躯干高度奖励 —— #
            K1 = 50.0
            upright_reward = np.exp(-K1 * (pos_z - upright_target_z) ** 2)

            # —— 2. 躯干 pitch “二次”奖励 —— #
            K2 = 10.0
            balance_reward = np.exp(-K2 * (pitch ** 2))

            # —— 3. 能量消耗惩罚 —— #
            K_e = 2.0
            energy_penalty = -K_e * np.sum(action**2)
            # energy_penalty = np.exp(-K_e * np.sum(action**2))

            # —— 4. 向前移动奖励 —— #
            K_forward2 = 10.0
            # forward_reward = K_forward2 * vel_x
            forward_reward = np.exp(-K_forward2 * (vel_x - 0.5) ** 2)

            fall_penalty = -50.0 if pos_z < 0.3 else 0.0

            reward = (upright_reward + balance_reward 
                      + fall_penalty + base_reward
                      + energy_penalty
                      + forward_reward)

        # # 基础奖励：
        # base_reward = 0.0625   #0.0625

        # # 主要奖励：向前移动速度
        # forward_reward = vel_x
        
        # # 惩罚：偏离中心线
        # # path_penalty = -self.path_weight * pos_y ** 2
        # path_penalty = -3.0 * pos_y ** 2

        # # 惩罚：躯干高度过低
        # height_penalty = -50 * (pos_z - 0.56) ** 2

        # # 惩罚：能量消耗
        # action_magnitude = np.sum(np.square(action))
        # # energy_penalty = -self.energy_weight * action_magnitude
        # energy_penalty = -0.02 * action_magnitude
        
        # # 惩罚：摔倒
        # fall_penalty = 0 if pos_z >= 0.3 else -5  # 如果躯干高度过低，认为摔倒
        
        # # 总奖励
        # reward = base_reward + forward_reward + fall_penalty #+ path_penalty + energy_penalty + height_penalty 
        
        return reward
    
    def _check_done(self, observation):
        pos_z = observation[1]  # 躯干z方向位置
        return pos_z < 0.3  # 如果躯干高度过低，认为摔倒，结束当前episode
    
    def close(self):
        p.disconnect()