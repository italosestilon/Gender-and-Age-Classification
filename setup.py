from setuptools import setup, find_packages

setup(name='cifar10-keras',
    version='2.0',
    packages=find_packages(),
    description='example to run keras on gcloud ml-engine',
    author='Fuyang Liu',
    author_email='fuyang.liu@example.com',
    license='MIT',
    install_requires=[
    'keras',
    'h5py'
    ],
    zip_safe=False)