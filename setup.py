#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

with open('bayes/__init__.py') as fid:
    for line in fid:
        if line.startswith('__version__'):
            VERSION = line.strip().split()[-1][1:-1]
            break

with open('requirements.txt') as fid:
    INSTALL_REQUIRES = [l.strip() for l in fid.readlines() if l]

readme = open('README.rst').read()
doclink = """
Documentation
-------------
The full documentation is at http://bayes.rtfd.org."""
#history = open('HISTORY.rst').read().replace('.. :changelog:', '')

VERSION = '0.1.2'

setup(
    name='bayes-variants',
    version=VERSION,
    description='Bayes',
    long_description=readme + '\n\n' + doclink + '\n\n',  #+ history,
    author='Krzysztof Joachimiak',
    # author_email='',
    url='https://github.com/krzjoa/Bayes',
    packages=find_packages(where='.', exclude=('tests')),
    package_dir={'bayes': 'bayes'},
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    license='MIT',
    zip_safe=False,
    keywords='bayes',
    classifiers=[
        'Programming Language :: Python',
        'License :: OSI Approved :: MIT License',
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
    ],
)
