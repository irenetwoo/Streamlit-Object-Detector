import os
import sys
import setuptools
from setuptools.command.install import install

VERSION = "1.10.0"  # PEP-440
NAME = "streamlit"

INSTALL_REQUIRES = [
    "gcc7==0.0.7" , "opencv-python-headless==4.5.2.52", "pycocotools==2.0.2", "torch==1.8.1", 
    "torchvision==0.13.0", "torchaudio==0.16.0",
]

setuptools.setup(
        name=NAME,
    version=VERSION,
    project_urls={
        "Source": "https://github.com/irenetwoo/Streamlit-Object-Detector",
    },
    license="Apache 2",
    package_data={"streamlit": ["py.typed", "hello/**/*.py"]},
    packages=setuptools.find_packages(exclude=["tests", "tests.*"]),  
    # Requirements
    install_requires=INSTALL_REQUIRES,
    zip_safe=False,
    entry_points={"console_scripts": ["streamlit = streamlit.web.cli:main"]},
    scripts=["bin/streamlit.cmd"],
    cmdclass={
        "verify": VerifyVersionCommand,
    },
)
