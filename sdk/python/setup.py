"""
AetherMind Python SDK Setup
Enables easy integration of AetherMind AGI into any Python application
"""

from setuptools import setup, find_packages
import os

# Read the long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from package
with open("aethermind/__init__.py", "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

setup(
    name="aethermind",
    version=version,
    author="AetherMind Team",
    author_email="dev@aethermind.ai",
    description="Official Python SDK for AetherMind AGI - Real AGI, Not Role-Playing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/United-Visions/AetherAGI",
    project_urls={
        "Bug Tracker": "https://github.com/United-Visions/AetherAGI/issues",
        "Documentation": "https://aethermind.ai/documentation",
        "Source Code": "https://github.com/United-Visions/AetherAGI",
        "Discord": "https://discord.gg/aethermind",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "httpx>=0.24.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "async": [
            "aiohttp>=3.8.0",
        ],
    },
    keywords="agi ai artificial-intelligence machine-learning sdk api aethermind",
    license="Apache-2.0",
    include_package_data=True,
    zip_safe=False,
)
