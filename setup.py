#!/usr/bin/env python
import platform

import setuptools

if (
    platform.processor() == "arm" or platform.processor() == "i386"
) and platform.system() == "Darwin":
    tensorflow_os = "tensorflow-macos>=2.10.0,<=2.15.0"
else:
    tensorflow_os = "tensorflow>=2.12.0,<=2.15.0"

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="decimer",
    version="2.7.0",
    author="Kohulan Rajan",
    author_email="kohulan.rajan@uni-jena.de",
    maintainer="Kohulan Rajan, Otto Brinkhaus ",
    maintainer_email="kohulan.rajan@uni-jena.de, otto.brinkhaus@uni-jena.de",
    description="DECIMER 2.6.0: Deep Learning for Chemical Image Recognition using Efficient-Net V2 + Transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": ["decimer = DECIMER.decimer:main"],
    },
    url="https://github.com/Kohulan/DECIMER-Image_Transformer",
    packages=setuptools.find_packages(),
    license="MIT",
    install_requires=[
        tensorflow_os,
        "opencv-python",
        "pystow",
        "pillow-heif",
        "efficientnet",
        "selfies",
        "pyyaml",
    ],
    package_data={"DECIMER": ["repack/*.*", "efficientnetv2/*.*", "Utils/*.*"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
)
