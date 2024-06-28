from setuptools import setup

with open("requirements.txt") as f:
    require = [x.strip() for x in f.readlines()]

setup(
    name="timeinf",
    version="0.0",
    packages=["timeinf"],
    install_requires=require,
)
