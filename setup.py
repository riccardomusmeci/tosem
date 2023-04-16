import os
import re

from setuptools import find_packages, setup

# get version string from module
with open(
    os.path.join(os.path.dirname(__file__), "src", "__init__.py"),
    "r",
) as f:
    pattern = r"__version__ = ['\"]([^'\"]*)['\"]"
    version = re.search(pattern, f.read(), re.M).group(1)  # type: ignore

# Get package dependencies from requirement files
with open(os.path.join("requirements.txt"), "r") as fin:
    reqs = fin.readlines()
with open("test-requirements.txt", "r") as fin:
    test_reqs = fin.readlines()
with open("dev-requirements.txt", "r") as fin:
    dev_reqs = fin.readlines()
with open("doc-requirements.txt", "r") as fin:
    doc_reqs = fin.readlines()

setup(
    name="tosem",
    version=version,
    author="Riccardo Musmeci",
    description="PyTorch Semantic Segmentation library to train and test your models",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    packages=find_packages("tosem"),
    package_dir={"": "src"},
    package_data={"src": ["py.typed"]},
    install_requires=reqs,
    extras_require={
        "dev": dev_reqs + test_reqs + doc_reqs + reqs,
        "test": test_reqs,
        "doc": doc_reqs,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8, <=3.11",
    zip_safe=False,
)
