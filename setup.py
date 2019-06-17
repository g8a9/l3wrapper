import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="l3wrapper",
    version="0.1.0",
    url="https://gitlab.com/g8a9/l3wrapper",
    license='MIT',

    author="Giuseppe Attanasio",
    author_email="giuseppe.attanasio@polito.it",

    description="A simple Python 3 wrapper around L3 binaries.",
    long_description=read("README.rst"),
    long_description_content_type="text/x-rst",

    packages=find_packages(exclude=('tests',)),

    install_requires=['pandas'],

    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
