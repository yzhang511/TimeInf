from setuptools import setup

with open("requirements.txt") as f:
    require = [x.strip() for x in f.readlines()]

setup(
    name="time_series_influences",
    version="0.1",
    packages=["time_series_influences"],
    install_requires=require,
)
