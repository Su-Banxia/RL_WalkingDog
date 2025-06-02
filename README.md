`simple_test.py`  是测试pybullet库的

`robot_env.py`  包括模型输入和输出定义，奖励函数设置等（奖励函数得多改改）

`ddpg.py` 是用DDPG方法更新actor和critic两个网络

`start_train.py` 就是用来训练的，可以在终端用

```
python start_train.py --train --timesteps 100000
```

开始训练，其他参数也可以设置。

训练完毕后，用`test_robot.py`在pybullet中运行查看效果