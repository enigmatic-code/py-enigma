# this file allows enigma.py to be installed directly from GitHub
#
# using Poetry, add the following in pyproject.toml file:
#
# --
# [tool.poetry.dependencies.enigma]
# git = "https://github.com/enigmatic-code/py-enigma.git"
# --
#
#
# [[[ Formerly adding this to requirements.txt worked:
# --
# git+https://github.com/enigmatic-code/py-enigma.git#egg=enigma
# --
# but not any more. ]]]


from setuptools import setup

# minimal setup config
setup(name='enigma', py_modules=['enigma'])
