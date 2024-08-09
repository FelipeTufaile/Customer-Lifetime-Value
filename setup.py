from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('version.txt') as v:
      version = v.read()

setup(name='customer_ecomomics',
      version=version,
      description='Customer lifetime value model that uses a zero-inflated mean squared error loss with a deep neural network framework',
      url='https://github.com/FelipeTufaile/Customer-Lifetime-Value.git',
      author='Felipe Solla Tufaile',
      author_email='f.tufaile@gmail.com',
      license='The MIT License (MIT)',
      packages=find_packages(),
      include_package_data=True,
      python_requires='>=3.7',
      install_requires=requirements
)