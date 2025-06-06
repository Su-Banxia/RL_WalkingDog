`robot_env.py`  包括模型输入和输出定义，奖励函数设置等（奖励函数得多改改）

`td3.py` 是用TD3算法更新actor和critic1，critic2三个网络

`start_train_td3.py` 就是用来训练的，可以在终端用

```
python start_train_td3.py --train --timesteps 100000
```

开始训练，其他参数也可以设置。

训练完毕后，用`test_robot.py`在pybullet中运行查看效果