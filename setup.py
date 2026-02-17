from setuptools import setup, find_packages

setup(
    name="kronos",
    version="0.1",
    packages=find_packages(), # 这会自动寻找文件夹下带 __init__.py 的目录
)