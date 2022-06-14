# Copyright 2022 Quantinuum (www.quantinuum.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    'numpy>=1.20.1',
    'scipy>=1.6.1',
    'matplotlib>=3.5.1',
    'pandas>=1.2.3'
]

setuptools.setup(
    name="qtm_spec",
    version='0.1.1',
    author="Quantinuum",
    author_email="charlie.baldwin@quantinuum.com",
    license="Apache 2.0",
    description="Analyze Quantinuum Specification Data",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="",
    packages=setuptools.find_namespace_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    include_package_data=True,
    keywords="quantum computing",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering"
    ],
    zip_safe=False,
)
