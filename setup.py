"""
Setup script for the Voice Changer Platform.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="voice-changer-platform",
    version="1.0.0",
    author="Voice Changer Platform",
    description="Real-time AI Voice Changer using RVC for Windows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "voice-changer=src.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Multimedia :: Sound/Audio",
    ],
    include_package_data=True,
    package_data={
        "": ["config/**/*.json", "models/.gitkeep"],
    },
)
