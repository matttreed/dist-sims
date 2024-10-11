from setuptools import setup, find_packages

setup(
    name="wash-sim",
    version="0.1.0",
    description="A brief description of your module",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "torchtext",
        "portalocker>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
