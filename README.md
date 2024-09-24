# 📖 tag-recommender
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Tumblr.svg/1920px-Tumblr.svg.png" alt="drawing" width="200"/>


## 📝 Overview
This repository contains the code for a tag recommender system that recommends tags
for Tumblr posts based on previous hashtags used in the post.
The system uses a combination of text processing techniques, statistical methods and
machine learning models to predict the most relevant tags for a given post.

## 📚 Getting Started

This guide provides instructions for setting up your Python environment,
including installing Python 3.11, Poetry,
and managing dependencies with a virtual environment.


## 📑 Table of Contents
1. [🔧 Install Python 3.11](#-install-python-311)
2. [📦 Install Poetry](#-install-poetry)
3. [🌱 Create a Virtual Environment with Poetry](#-create-a-virtual-environment-with-poetry)
4. [📥 Install Existing Dependencies](#-install-existing-dependencies)
5. [🔗 Download Data](#-download-data)
6. [📘 Open Jupyter Notebook](#-open-jupyter-notebook)
7. [📂 Project Directory Structure](#-project-directory-structure)

## 🔧 Install Python 3.11

To install Python 3.11, visit the [official Python download page](https://www.python.org/downloads/release/python-3110/) and follow the instructions for your operating system.

## 📦 Install Poetry
Poetry is a tool for dependency management and packaging in Python.
(alternative to venv and to pipenv)
To install Poetry, visit the [official Poetry installation page](https://python-poetry.org/docs/#installation).

## 🌱 Create a Virtual Environment with Poetry
After installing Poetry, you can create a virtual environment for your project 
by navigating to your project directory and running:

```bash
poetry init  # Initializes a new pyproject.toml file
poetry install  # Creates a virtual environment and installs dependencies
```

To activate the virtual environment, run: 
```bash
poetry shell
```

## 📥 Install Existing Dependencies
To install the projects dependencies using the `pyproject.toml` and `poetry.lock` file, 
run the following command:
```bash 
poetry install
```