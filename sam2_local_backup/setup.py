from setuptools import setup, find_packages

setup(
    name="sam2-local",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.3.1",
        "torchvision>=0.18.1",
        "hydra-core>=1.3.2",
        "iopath>=0.0.24",
        "timm>=0.9.12",
        "opencv-python>=4.7.0",
        "matplotlib>=3.0.0",
        "pillow>=9.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
    python_requires=">=3.10",
)
