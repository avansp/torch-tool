from setuptools import setup, find_packages

setup(
    name='torch-tool',
    author='Avan Suinesiaputra',
    author_email='avan.sp@gmail.com',
    version='0.1.0',
    license='LICENSE',
    description='My collections of pytorch models',
    packages=find_packages(),
    long_description=open('README.md').read(),
)