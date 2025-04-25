FROM osrf/ros:noetic-desktop AS rosa-ros1
LABEL authors="Rob Royce"

ENV DEBIAN_FRONTEND=noninteractive
ENV HEADLESS=false
ARG DEVELOPMENT=false

# Install linux packages
RUN apt-get update && apt-get install -y \
    ros-$(rosversion -d)-turtlesim \
    locales \
    xvfb \
    python3.9 \
    python3-pip \
    python3.9-distutils

# Install packages for system Python and Python 3.9
RUN python3 -m pip install --no-cache-dir -U pip && \
    python3 -m pip install --no-cache-dir -U python-dotenv catkin_tools distro && \
    python3.9 -m pip install --no-cache-dir -U pip && \
    python3.9 -m pip install --no-cache-dir -U python-dotenv && \
    python3.9 -m pip install --no-cache-dir -U \
        "pydantic<2" \
        "langchain<0.2" \
        "langchain-community<0.1" \
        "langchain-core<0.2" \
        "langchain-openai<0.2" \
        pyinputplus rich "numpy<1.25"

RUN rosdep update && \
    echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "export ROSLAUNCH_SSH_UNKNOWN=1" >> /root/.bashrc

# --- Workspace for Dependencies (/app_deps) ---
WORKDIR /app_deps
RUN mkdir src build devel logs
COPY deps_for_docker/global_path_planning /app_deps/src/global_path_planning

# Build the dependency workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    cd /app_deps && \
    catkin init && \
    echo \"Building dependency workspace (/app_deps)...\" && \
    catkin build -v && \
    echo \"Dependency workspace build complete.\""

# --- Workspace for Main Agent (/app) ---
WORKDIR /app
RUN mkdir src build devel logs
COPY src/rosa /app/src/rosa
COPY src/turtle_agent /app/src/turtle_agent
# COPY setup.py /app/setup.py                  # Copy top-level setup.py if needed for rosa package

# Build the main workspace, EXTENDING the dependency workspace
RUN /bin/bash -c "echo \"Setting CMAKE_PREFIX_PATH for main build...\" && \
    export CMAKE_PREFIX_PATH=/app_deps/devel:$CMAKE_PREFIX_PATH && \
    source /opt/ros/noetic/setup.bash && \
    cd /app && \
    catkin init && \
    echo \"Cleaning main workspace before build...\" && \
    catkin clean -y && \
    echo \"Building main workspace (/app) with explicit CMAKE_PREFIX_PATH...\" && \
    catkin build -v && \
    echo \"Main workspace build complete.\""

# --- Aliases and CMD (Source the FINAL devel space: /app/devel) ---
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    # Start alias sources the final workspace /app/devel
    echo "alias start='source /app/devel/setup.bash && roslaunch turtle_agent agent.launch'" >> /root/.bashrc && \
    echo "export ROSLAUNCH_SSH_UNKNOWN=1" >> /root/.bashrc

# CMD sources the final workspace /app/devel
CMD ["/bin/bash", "-c", "source /app/devel/setup.bash && echo 'Workspaces built and sourced. Run start to launch.' && /bin/bash"]
