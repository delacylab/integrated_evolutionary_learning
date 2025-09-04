from setuptools import setup, find_packages
from pathlib import Path
README = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="integrated-evolutionary-learning",          # pip distribution name
    version="0.1.0",
    description="Integrated Evolutionary Learning (IEL) utilities and models",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Wai-yin Lam",
    author_email="u6054998@utah.edu",
    url="https://github.com/delacylab/integrated_evolutionary_learning",
    license="Apache-2.0",
    python_requires=">=3.9",
    packages=find_packages(include=["IEL*"]),
    include_package_data=True,
    install_requires=[
        "numpy>=1.24.4",
        "scipy>=1.11.0",
        "pandas>=2.2.2",
        "scikit-learn>=1.5.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.12.2",
        "imbalanced-learn>=0.12.3",
        "captum>=0.7.0",
        "pillow>=10.0.1",
        "statsmodels>=0.14.1",
        "eli5>=0.13.0",
        "kneed>=0.8.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)