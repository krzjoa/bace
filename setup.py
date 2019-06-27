#!/usr/bin/env python

from setuptools import setup, find_packages


#if sys.argv[-1] == 'publish':
#    os.system('python setup.py sdist upload')
#    sys.exit()

with open('bace/__init__.py') as fid:
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
The full documentation is at http://bace.rtfd.org."""

VERSION = '1.0.0'

setup(
    name='bace',
    version=VERSION,
    description='bace',
    long_description=readme + '\n\n' + doclink + '\n\n',
    author='Krzysztof Joachimiak',
    url='https://github.com/krzjoa/bace',
    packages=find_packages(where='.', exclude=('tests')),
    package_dir={'bace': 'bace'},
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
