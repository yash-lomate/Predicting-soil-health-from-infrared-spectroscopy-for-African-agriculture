"""
Setup script for the soil property prediction package.
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='soil-property-prediction',
    version='1.0.0',
    author='Yash Lomate',
    description='Multi-target regression for predicting soil properties from infrared spectroscopy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yash-lomate/Predicting-soil-health-from-infrared-spectroscopy-for-African-agriculture',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'soil-train=train:main',
            'soil-predict=predict:main',
        ],
    },
)
