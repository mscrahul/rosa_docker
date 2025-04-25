cmake_minimum_required(VERSION 3.0.2)
project(rosa)

find_package(catkin REQUIRED COMPONENTS
  # Add components rosa depends on, e.g.:
  # rospy
  # std_msgs
)

# Declare that this package uses Python and needs setup.py to be run
catkin_python_setup()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES rosa
   CATKIN_DEPENDS rospy # Add ROS dependencies rosa exports or uses heavily
#  DEPENDS system_lib
)

# No install rules needed here if setup.py handles it