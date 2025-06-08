`robot_env.py` - Contains model input/output definitions and reward function settings (reward function needs more tuning)

`td3.py` - Implements TD3 algorithm for updating actor, critic1, and critic2 networks

`start_train_td3.py` - Used for training. Can be executed in terminal with:

```
python scripts\start_train_td3.py --train --timesteps 100000
```

to start training. Other parameters can also be configured.

After training completes, use test_robot.py to visualize the results in PyBullet