#!/usr/bin/env python3
from setuptools import setup

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

install_reqs = parse_requirements('requirements.txt')
# reqs = [str(ir.req) for ir in install_reqs]


setup(
    name='my_Structure_from_Motion',
    version='0.1',
    install_requires=install_reqs
)
