from setuptools import setup, find_packages

setup(
    name='lightgbmmodeloptimizer',  # Name of the package
    version='0.0.1',      # Version number
    description='Reduce size and improve inference time of the trained lightgbm model',
    author='Rajat Goyal',
    readme = "README.md",
    packages=find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)