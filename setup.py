from pathlib import Path
from typing import List

import setuptools

# The directory containing this file
HERE = Path(__file__).parent.resolve()

# The text of the README file
NAME = 'gefest'
AUTHOR = 'NSS Lab'
SHORT_DESCRIPTION = 'The toolbox for the generative design of physical objects'
README = Path(HERE, 'README.rst').read_text()
URL = 'https://github.com/ITMO-NSS-team/GEFEST'
VERSION = '0.1.0'
REQUIRES_PYTHON = '>=3.9'
LICENSE = 'BSD 3-Clause'


def _readlines(*names: str, **kwargs) -> List[str]:
    encoding = kwargs.get('encoding', 'utf-8')
    lines = Path(__file__).parent.joinpath(*names).read_text(encoding=encoding).splitlines()
    return list(map(str.strip, lines))


def _extract_requirements(file_name: str):
    return [line for line in _readlines(file_name) if line and not line.startswith('#')]


def _get_requirements(req_name: str):
    requirements = _extract_requirements(req_name)
    return requirements


setuptools.setup(
    name=NAME,
    author=AUTHOR,
    author_email='itmo.nss.team@gmail.com',
    description=SHORT_DESCRIPTION,
    long_description=README,
    long_description_content_type='text/x-rst',
    url=URL,
    version=VERSION,
    python_requires=REQUIRES_PYTHON,
    license=LICENSE,
    packages=setuptools.find_packages(exclude=['test*']),
    package_data={'': ['*']},
    include_package_data=True,
    install_requires=_get_requirements('requirements.txt'),
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
