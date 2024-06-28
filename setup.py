from setuptools import setup,find_packages
from typing import List

hypen_e='-e .'
def get_packages(file_name)->List[str]:
    requirements=[]
    with open(file_name) as file:
        requirements=file.readlines()
        requirements= [req.replace("\n","") for req in requirements]
        if hypen_e in requirements:
            requirements.remove(hypen_e)

    return requirements


setup(
    name="StudentPerformanceIndicator",
    author="Ehetsham",
    author_email="ehetsham.s@gmail.com",
    version='0.1',
    install_requires=get_packages("requirements.txt"),
    packages=find_packages()
)