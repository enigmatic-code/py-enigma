# this file allows enigma.py to be installed directly from GitHub
#
# using pip:
# --
# pip3 install "enigma @ git+https://github.com/enigmatic-code/py-enigma"
# --
#
# using Poetry, add the following in pyproject.toml file:
# --
# [tool.poetry.dependencies.enigma]
# git = "https://github.com/enigmatic-code/py-enigma.git"
# --
#
#
# [[[
# Formerly adding this to requirements.txt worked:
# --
# git+https://github.com/enigmatic-code/py-enigma.git#egg=enigma
# --
# but not any more.
# ]]]

from setuptools import setup

# minimal setup config
setup(
  name='enigma',
  version='2.7.20230124',
  description='Useful routines for solving Enigma (and other) puzzles',
  author='Jim Randell',
  author_email='jim.randell@gmail.com',
  url='http://www.magwag.plus.com/jim/enigma.html',
  py_modules=['enigma'],
)
