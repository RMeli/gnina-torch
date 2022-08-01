import sys
from typing import Optional

from setuptools import find_packages, setup

import versioneer

short_description = "PyTorch implementation of GNINA scoring function".split("\n")[0]

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description: Optional[str] = handle.read()
except Exception:
    long_description = None


setup(
    # Self-descriptive entries which should always be present
    name="gninatorch",
    author="Rocco Meli",
    author_email="rocco.meli@biodtp.ox.ac.uk",
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="MIT",
    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),
    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,
    # Allows `setup.py test` to work correctly with pytest
    # setup_requires=[] + pytest_runner,
    setup_requires=pytest_runner,
    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # url='http://www.my_package.com',  # Website
    install_requires=[
        "torch",
        "molgrid",
        "numpy",
    ],
    extra_require={
        "all": [
            "pytorch-ignite",
            "scipy",
            "pandas",
            "scikit-learn",
            "tqdm",
            "mlflow",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-xdist>=2.5",
            "pytest-cov>=3.0",
        ],
        "doc": [
            "sphinx",
            "sphinx_rtd_theme",
        ],
    },
    platforms=[
        "Linux",
    ],  # molgrid only supports linux
    python_requires=">=3.7",  # Python version restrictions
    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,
)
