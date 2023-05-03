from setuptools import setup, find_packages

setup(
    name="simulatedmicroscopy",
    version="1.0.0",
    author="Roy Hoitink",
    author_email="L.D.Hoitink@uu.nl",
    long_description=open("README.md").read(),
    packages=find_packages(include=["simulatedmicroscopy", "simulatedmicroscopy.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "matplotlib",
        "h5py",
        "scipy",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "pytest",
            "pytest-cov",
            "bump2version",
        ],
        "docs": [
            "mkdocs",
            "mkdocs-material",
            "mkdocstrings[python]",
        ]
    },
)
