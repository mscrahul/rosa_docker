#  Copyright (c) 2024. Jet Propulsion Laboratory. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# Use generate_distutils_setup to integrate with Catkin
setup_args = generate_distutils_setup(
    packages=['rosa'],      # Declare the 'rosa' package
    package_dir={'': 'src'} # Tell setup the packages are in the 'src' directory
)

setup(**setup_args)
