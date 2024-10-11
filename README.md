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
3. [📥 Install Existing Dependencies](#-install-existing-dependencies)
4. [🔗 Download Data](#-download-data)
5. [📘 Open Jupyter Notebook](#-open-jupyter-notebook)
6. [📂 Project Directory Structure](#-project-directory-structure)

## 🔧 Install Python 3.11

To install Python 3.11, visit the [official Python download page](https://www.python.org/downloads/release/python-3110/) and follow the instructions for your operating system.

## 📦 Install Poetry
Poetry is a tool for dependency management and packaging in Python.
(alternative to venv and to pipenv)
To install Poetry, visit the [official Poetry installation page](https://python-poetry.org/docs/#installation).

## 📥 Install Repo Existing Dependencies
To install the projects dependencies using the `pyproject.toml` and `poetry.lock` file,
run the following command:

```bash
poetry install  # Creates a virtual environment and installs dependencies
```

To activate the virtual environment, run:
```bash
poetry shell
```

Duplicate the `.env.sample` file and name it as `.env` using the following command.
```bash
cp .env.sample .env
```

Export the environment variables:
```bash
export $(grep -v '^#' .env | xargs -0)
```

To split the data into training and testing sets, run the following command:
```bash
tag-recommender data split --input_file path/to/input_file --save_dir path/to/output_dir
```

Run the rest api locally by running the following command:
```bash
tag-recommender services run-rest
```
Documentation for the REST api can be found at `http://localhost:8000/docs`

Dockers can be built and run using the following commands:
From the top-level directory of the repository
```bash
docker build -t tag-recommender:latest -f docker/Dockerfile .
```
Run the Docker Container
```bash
docker run -p 8000:8000 tag-recommender:latest
```
