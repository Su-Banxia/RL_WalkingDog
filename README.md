## RL Part:

`robot_env.py` - Contains model input/output definitions and reward function settings (reward function needs more tuning)

`td3.py` - Implements TD3 algorithm for updating actor, critic1, and critic2 networks

`start_train_td3.py` - Used for training. Can be executed in terminal with:

```
python -m scripts.start_train_td3 --train --timesteps 1000000 --load_actor final_actor.pth --load_critic1 final_critic1.pth --load_critic2 final_critic2.pth
```

to start training. Other parameters can also be configured.

After training completes, use `test_robot.py` to visualize the results in PyBullet:

```
python -m scripts.test_robot
```
## 3DGS Resources

### Core Implementation
- [Windows Install Guide](https://github.com/jonstephens85/gaussian-splatting-Windows)  
  Simplified Windows setup for 3DGS

### Datasets
- [USTC Resource Pack](https://rec.ustc.edu.cn/share/aa59d8f0-4461-11f0-a898-03cb5c3e4c28)  
  Models & scene data

### Engine Integration
- [Unreal Engine 5 Plugin](https://github.com/xverse-engine/XV3DGS-UEPlugin)  
  Real-time 3DGS rendering for UE5
