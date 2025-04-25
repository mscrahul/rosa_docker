from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['global_path_planning'], # Ensure your package name is here
    package_dir={'': 'scripts'}        # Tells setup where to find the 'global_path_planning' directory relative to setup.py
)

setup(**setup_args)
