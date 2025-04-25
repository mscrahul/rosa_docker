# rosa
rosa and node inside docker. other outside
# ROSA Integration Attempt with Docker (Summary for Expert Review)

## Project Goal

The primary objective was to integrate the ROSA (Robot Operating System Agent) framework with a custom indoor mobile robot running ROS 1 Noetic. The specific aim was to enable navigation control via natural language by having ROSA call a custom ROS service, `/set_pixel_goal`, defined within the robot's existing `global_path_planning` Catkin package. This service accepts target pixel coordinates corresponding to a map used by the robot's navigation system.

## Chosen Approach: Docker for Isolation

Given that ROSA and its primary dependency, Langchain, typically require Python 3.9+, while ROS 1 Noetic officially targets Python 3.8 on Ubuntu 20.04, a Docker-based approach was chosen.

*   **Environment:** The ROSA agent (specifically, a modified version of the `turtle_agent` example) and its Python 3.9 dependencies (Langchain, Pydantic V1, OpenAI library, etc.) would run inside a Docker container based on `osrf/ros:noetic-desktop`.
*   **Host System:** The main robot code, including the ROS master (`roscore`) and the node providing the `/set_pixel_goal` service, would run directly on the host machine (Ubuntu 20.04, ROS Noetic, Python 3.8).
*   **Communication:** The Docker container was run with `--network host` to allow seamless ROS communication (topic subscriptions, service calls) between the agent inside the container and the ROS nodes running on the host.
*   **Dependency Management:** The core `rosa` library was intended to be installed via `pip` within the container. The custom service definition (`SetPixelGoal.srv`) needed to be made available to the agent's Python code. The strategy for this involved copying the source code of the `global_path_planning` package (containing the `.srv` file) into the Docker container's Catkin workspace (`/app/src`) and building it using `catkin build` during the Docker image creation process.

## Core Technical Challenge

The central difficulty revolved around correctly integrating the build artifacts and runtime environments of two distinct systems within the Docker container:

1.  **Catkin Build System:** Needed to compile the `.srv` file from `global_path_planning` into usable Python modules (`global_path_planning.srv`) located within the Catkin `devel` space (`/app/devel/lib/python3/dist-packages`).
2.  **Standard Python Environment:** Needed to run the `turtle_agent.py` script and import both the core `rosa` library (installed via `pip`) and the Catkin-generated service modules.

The primary challenge was ensuring that the Python environment sourced at runtime (`/app/devel/setup.bash`) correctly exposed *both* the Catkin-generated modules *and* the pip-installed `rosa` library.

## Implementation Steps and Errors Encountered

Several strategies were attempted within the `Dockerfile` to achieve a working build and runtime environment:

1.  **Initial State (Agent Runs, Service Call Fails):**
    *   **Setup:** `global_path_planning` source was copied, `rosa` library was pip-installed during build. `catkin build` likely only built `turtle_agent`.
    *   **Outcome:** The agent (`turtle_agent.py`) launched successfully. Basic interaction worked. `from rosa import ...` worked.
    *   **Error:** Attempting to use the `navigate_to_named_location` tool (which calls `/set_pixel_goal`) failed at runtime with `AttributeError: type object 'SetPixelGoal' has no attribute '_request_class'`.
    *   **Reason:** The Python code inside the container couldn't properly import the *generated* service definition because `global_path_planning` hadn't been correctly built *by Catkin* within the sourced environment.

2.  **Attempting Catkin Integration (Build All / Sequential / Overlay):** To fix the `AttributeError`, we focused on ensuring `catkin build` processed `global_path_planning` correctly and that `turtle_agent` depended on it.
    *   **Modifications:** Added `global_path_planning` as a dependency in `turtle_agent`'s `package.xml` and `CMakeLists.txt`. Tried various `catkin build` strategies in the Dockerfile (`build all`, build dependency first then agent, build in separate overlayed workspaces). Also tried treating the core `rosa` library as a Catkin package.
    *   **Build Errors Encountered:**
        *   `Error: Given package 'global_path_planning' is not in the workspace...`: This occurred when trying to build `global_path_planning` by name explicitly, indicating Catkin wasn't discovering it correctly at that stage (later resolved by removing `CATKIN_IGNORE` and ensuring `package.xml` was present/valid).
        *   `CMake Error at /opt/ros/noetic/share/catkin/cmake/catkinConfig.cmake:83 (find_package): Could not find a package configuration file provided by "global_path_planning"...`: **This became the persistent error.** Even when `catkin build` found all packages and successfully built `global_path_planning` first (verified by "X of Y packages succeeded" messages), the subsequent CMake configuration step for `turtle_agent` failed because it could not locate the `global_path_planningConfig.cmake` file via `find_package`. This occurred despite trying single workspace builds, sequential builds with intermediate sourcing, and overlayed workspace builds with explicit `CMAKE_PREFIX_PATH` adjustments.
    *   **Runtime Errors Encountered (During Build Debugging):**
        *   `ModuleNotFoundError: No module named 'dotenv'`: Fixed by explicitly installing `python-dotenv` for `python3.9` using pip in the Dockerfile.
        *   `ModuleNotFoundError: No module named 'rosa'`: This reappeared *after* we successfully got Catkin to build `global_path_planning`. It indicated that sourcing the complete `devel/setup.bash` (now containing info for `global_path_planning` and `turtle_agent`) was preventing Python from finding the `rosa` library installed elsewhere by `pip`. Attempts to fix this by adjusting `PYTHONPATH` in the `start` alias or `CMD` were unsuccessful in reliably resolving the import within the `roslaunch` environment. Treating `rosa` as a Catkin package also ultimately led back to the persistent `find_package` error for `global_path_planning`.
        *   
WhatsApp Image 2025-04-26 at 12.47.09 AM.jpeg


## Final State & Sticking Point (Docker Approach)

The Docker build process consistently fails during the CMake configuration phase for the `turtle_agent` package. Despite successfully building the `global_path_planning` dependency in a preceding step (or within the same build invocation) and ensuring the `CMAKE_PREFIX_PATH` appears to include the dependency's `devel` space, the `find_package(catkin REQUIRED COMPONENTS global_path_planning)` call within `turtle_agent/CMakeLists.txt` fails to locate the necessary `global_path_planningConfig.cmake` file.

This prevents the successful build of the `turtle_agent` package within the Docker container when it has a declared dependency on `global_path_planning`. The root cause appears to be a failure in how the CMake environment or Catkin's internal state is propagated between the building of the dependency and the configuration of the dependent package within this specific Docker build setup. Standard methods like overlay workspaces and explicit environment sourcing did not resolve this specific `find_package` failure.
