# End-effector command: target pose + gripper opening.
# Pose is expressed in `header.frame_id` (typically "world").
# `gripper` is the target gripper joint position in metres,
# gripper range is [-0.003, 0.05]).
#
# Wire-compatible with anvil-workcell's copy of this message — keep both in
# sync. Real hardware publishes/subscribes this exact shape on
# /commanded_ee_{arm} and /ee_pose_{arm}.
std_msgs/Header header
geometry_msgs/Pose pose
float64 gripper
