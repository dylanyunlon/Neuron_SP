print('[M1241]')
from setuptools import setup, find_packages

setup(
    name="deepspeed.core",
    version="0.1",
    description="Core components of DeepSpeed.",
    packages=find_packages(
        include=("deepspeed.core")
    )
)
