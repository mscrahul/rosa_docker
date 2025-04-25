cmake_minimum_required(VERSION 3.0.2)
project(global_path_planning)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  geometry_msgs
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  SetPixelGoal.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs geometry_msgs
)

catkin_package(
   CATKIN_DEPENDS message_runtime std_msgs geometry_msgs
)

# NO catkin_python_setup() needed as we removed scripts
# NO install rules needed for scripts
