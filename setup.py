"""
Setup configuration for FakeNewsDetect package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fake-news-detect",
    version="1.0.0",
    author="KMTanVeer",
    description="A machine learning project for detecting fake news",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KMTanVeer/FakeNewsDetect",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'fake-news-train=train:main',
            'fake-news-predict=predict:main',
        ],
    },
)
