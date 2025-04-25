# ~/catkin_ws/src/rosa/src/turtle_agent/setup.py
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# Minimal setup just to enable catkin_python_setup()
setup_args = generate_distutils_setup(
    packages=[], # No packages declared here for installation via setup.py
    package_dir={'': 'scripts'}
)
setup(**setup_args)