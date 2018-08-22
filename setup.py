# this file allows enigma.py to be installed directly from GitHub
#
# using the following requirements.txt file:
#
# --
# git+https://github.com/enigmatic-code/py-enigma.git#egg=enigma
# --

from distutils.core import setup

setup(
  name='enigma',
  #version='0.0', # dummy version for pip
  py_modules=['enigma'],
)
