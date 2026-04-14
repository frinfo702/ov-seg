from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(path):
    requirements = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)
    return requirements

setup(
    name="clip",
    py_modules=["clip"],
    version="1.0",
    description="",
    author="OpenAI",
    packages=find_packages(exclude=["tests*"]),
    install_requires=read_requirements(Path(__file__).with_name("requirements.txt")),
    include_package_data=True,
    extras_require={"dev": ["pytest"]},
)
