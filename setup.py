from setuptools import setup, find_packages

setup(
    name="rank_llms",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "anthropic",
        "ollama",
        "typer",
        "pydantic",
        "rich",
        "pandas",
    ],
    entry_points={
        "console_scripts": [
            "rank-llms=rank_llms.main:app",
        ],
    },
)