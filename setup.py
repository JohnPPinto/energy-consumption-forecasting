from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """
    This function reads the requirements.txt file and returns a list 
    containing all the libraries.

    Parameters
    ----------
    file_path : str
        A string containing the relative path to the requirements.txt 
        file.
    
    Returns
    -------
    requirements : list
        A list containing all the libraries for installation.
    """
    EDIT_FLAG = '-e .'
    requirements = []

    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [lib.replace('\n', '') for lib in requirements]

        if EDIT_FLAG in requirements:
            requirements.remove(EDIT_FLAG)

    return requirements

setup(name='energy_consumption_forecasting',
      version='0.0.1',
      author='John Pinto',
      author_email='pintojohn47@gmail.com',
      packages=find_packages(),
      install_requires=get_requirements('requirements.txt'))
