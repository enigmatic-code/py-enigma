# this file allows enigma.py to be installed directly from GitHub
#
#
# [install using pip and requirements.txt]
# add the following line to a requirements.txt file:
# --
# enigma @ git+https://github.com/enigmatic-code/py-enigma
# --
# and then run the following command to install/upgrade:
# --
# % pip3 install -U -r requirements.txt
# --
#
#
# [install/upgrade directly using pip (with git)]
# --
# % pip3 install -U "enigma @ git+https://github.com/enigmatic-code/py-enigma"
# --
#
#
# [install/upgrade directly using pip (without git)]
# --
# % pip3 install -U "enigma @ https://github.com/enigmatic-code/py-enigma/tarball/master"
# --
#
#
# [using Poetry]
# add the following in pyproject.toml file:
# --
# [tool.poetry.dependencies.enigma]
# git = "https://github.com/enigmatic-code/py-enigma.git"

from setuptools import setup

setup(
  name='enigma',
  version='2.7.20230213',
  description='Useful routines for solving New Scientist Enigma (and other) puzzles',
  author='Jim Randell',
  author_email='jim.randell@gmail.com',
  url='http://www.magwag.plus.com/jim/enigma.html',
  py_modules=['enigma'],
)
