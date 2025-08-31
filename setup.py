"""
RocketAPI Setup Configuration
Installation simple et dépendances minimales - approche ILN
"""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rocketapi",
    version="0.1.0",
    author="Anzize Daouda",
    author_email="contact@rocketapi.dev",
    description="FastAPI syntax with ILN superpowers - Revolutionary performance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anzize/rocketapi",
    project_urls={
        "Bug Tracker": "https://github.com/anzize/rocketapi/issues",
        "Documentation": "https://rocketapi.dev/docs",
        "Source Code": "https://github.com/anzize/rocketapi",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Framework :: AsyncIO",
        "Framework :: FastAPI",
    ],
    package_data={
        "rocketapi": ["py.typed"],
    },
)_dir={"": "."},
    packages=["rocketapi"],
    py_modules=["rocketapi"],
    python_requires=">=3.8",
    install_requires=[
        # Dépendances minimales - philosophie ILN
        "typing-extensions>=4.0.0",  # Pour compatibilité Python 3.8+
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.18.0", 
            "pytest-benchmark>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "testing": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.18.0",
            "pytest-benchmark>=4.0.0",
            "httpx>=0.23.0",  # Pour tests HTTP
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=8.0.0", 
            "mkdocstrings>=0.18.0",
        ],
        "performance": [
            "uvloop>=0.16.0",  # Faster event loop
            "orjson>=3.6.0",   # Faster JSON
            "cython>=0.29.0",  # Compilation speedup
        ]
    },
    entry_points={
        "console_scripts": [
            "rocketapi=rocketapi:main",
        ],
    },
    keywords=[
        "api", "fastapi", "web", "framework", "async", "performance", 
        "iln", "multi-paradigm", "concurrency", "optimization",
        "go", "rust", "javascript", "essences", "primitives"
    ],
    zip_safe=False,
    include_package_data=True,
    package