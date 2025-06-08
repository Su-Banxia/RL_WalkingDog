import bpy
import json
import math
from mathutils import Quaternion, Vector

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

robot_fbx_path = r"assets/dog.fbx" # FBX model file
bpy.ops.import_scene.fbx(filepath=robot_fbx_path)

json_path = r"assets/robot_motion.json" # JSON motion data
with open(json_path, 'r') as f:
    motion_data = json.load(f)

frame_rate = 30
bpy.context.scene.render.fps = frame_rate
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = len(motion_data["timesteps"])

for frame_idx, (positions, orientations) in enumerate(zip(
        motion_data["link_positions"],
        motion_data["link_orientations"]
)):
    bpy.context.scene.frame_set(frame_idx)

    for link_idx, (pos, orn) in enumerate(zip(positions, orientations)):
        if link_idx==0:obj_name="front_left_leg"
        elif link_idx==1:obj_name="front_right_leg"
        elif link_idx==2:obj_name="rear_left_leg"
        elif link_idx==3:obj_name="rear_right_leg"
        elif link_idx==4:obj_name="torso"
        if obj_name in bpy.data.objects:
            obj = bpy.data.objects[obj_name]

            obj.location = Vector(pos)
            obj.keyframe_insert(data_path="location")

            q = Quaternion((orn[3], orn[0], orn[1], orn[2]))  # w,x,y,z
            obj.rotation_mode = 'QUATERNION'
            obj.rotation_quaternion = q
            obj.keyframe_insert(data_path="rotation_quaternion")


bpy.ops.object.select_all(action='SELECT')
output_fbx = r"D:\project\pythonProject\.venv\else\RL_WalkingDog-main\robot_animation.fbx"
bpy.ops.export_scene.fbx(
    filepath=output_fbx,
    use_selection=True,
    bake_anim=True,
    bake_anim_use_all_bones=True,
    bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=False,
    add_leaf_bones=False,
    bake_anim_step=1,
    bake_anim_simplify_factor=0
)
