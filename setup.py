from setuptools import setup, find_packages

setup(
    name='lightgbmmodeloptimizer',  # Name of the package
    version='0.1.0',      # Version number
    description='Reduce size and improve inference time of the trained lightgbm model',
    readme = "README.md",
    packages=find_packages(),
    install_requires=[
        'lightgbm>=3.0.0,<=4.4.0',  # This allows any version within the specified range
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)