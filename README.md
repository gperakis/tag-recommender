# 📖 tag-recommender
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Tumblr.svg/1920px-Tumblr.svg.png" alt="drawing" width="200"/>


## 📝 Overview
This repository contains the code for a tag recommender system
that recommends tags for Tumblr posts based on previous hashtags used in the post.
The system uses a combination of text processing techniques, statistical methods,
and machine learning models to predict the most relevant tags for a given post.

## 📚 Getting Started
This guide provides instructions for setting up your Python environment,
installing dependencies, downloading data, and running the application.

## 📑 Table of Contents
1. [🔧 Install Python 3.11](#-install-python-311)
2. [📦 Install Poetry](#-install-poetry)
3. [📥 Install Existing Dependencies](#-install-existing-dependencies)
4. [📂 Data](#-data)
5. [💻 Running the Application](#-running-the-application)˚
6. [📘 Open Jupyter Notebook](#-open-jupyter-notebook)
7. [🐳 Docker Setup](#-docker-setup)
8. [📂 Project Directory Structure](#-project-directory-structure)

## 🔧 Install Python 3.11
To install Python 3.11,
visit the [official Python download page](https://www.python.org/downloads/release/python-3110/)
and follow the instructions for your operating system.

## 📦 Install Poetry
Poetry is a tool for dependency management and packaging in Python.
It provides a streamlined approach to managing environments and dependencies.
To install Poetry,
visit the [official Poetry installation page](https://python-poetry.org/docs/#installation).

## 📥 Install Repo Existing Dependencies
To install the project's dependencies using the `pyproject.toml` and `poetry.lock` file,
run the following command:
```bash
poetry install  # Creates a virtual environment and installs dependencies
```

To install the dev dependencies, run:
```bash
poetry install --dev
```


To activate the virtual environment, run:
```bash
poetry shell
```

Duplicate the `.env.sample` file and name it `.env`:
```bash
cp .env.sample .env
```

Export the environment variables:
```bash
export $(grep -v '^#' .env | xargs -0)
```

## 📂 Data
The dataset should be located in the `data/` folder and should be a CSV file.

To split the data into training and testing sets, use the following command:
```bash
tag-recommender data split --input_file path/to/input_file --save_dir path/to/output_dir
```

If you wish to see all the provided training methodologies simply run
```bash
tag-recommender models
```

If you wish to train you own co-occurrence model, use the following command.
This will train the model using the settings found in the `config.py` file
(csm settings).
```bash
tag-recommender models train-co-occurrence
```

To train the Tag2Vec model use the following command:
```bash
tag-recommender models train-tag2vec
```

To train the Association Rule model use the following command:
```bash
tag-recommender models train-tag-rules
```

To train a Triplet Loss model use the following command to see
the available options
```bash
tag-recommender models train-triplet-tag-model --help
```


Alternatively you may download the pre-trained models from from the following
[Google Drive](https://drive.google.com/drive/folders/1XMv8aQweLvoNxaaA-o9ChH1D19zlB31f?usp=sharing):

The models should be saved in the `artifacts/models` directory.

## 💻 Running the Application

To run the REST API locally, execute the following command:
```bash
tag-recommender services run-rest
```
API documentation will be available at http://localhost:8000/docs.

To run a `stress test` on the API, execute the following command:
```bash
tag-recommender services run-rest-stress-test --help
```

There also exists a `streamlit application` where you can interact with the model.
To run the app you first need to ensure you have installed the dev dependencies.
Run the following command to start the application:
```bash
streamlit run tag_recommender/app.py
```
The app looks like in the following image
![streamlit app](![img.png](images/streamlit_app.png)

## 📘 Open Jupyter Notebook
To open and explore the notebooks, follow these steps:
```bash
jupyter notebook
```
Navigate to the notebooks/ directory where you will find the following notebooks:
- `1_tags_eda.ipynb`: Initial exploratory data analysis on tags.
- `2_split_dataset.ipynb`: Script for splitting the dataset.
- `3_hashtags_normalization.ipynb`: Steps for normalizing hashtags.
- `4_frequent_patterns.ipynb`: Notebook focusing on frequent pattern mining using MLextend.
- `5_frequent_patterns_pyspark.ipynb`: Frequent pattern mining using PySpark.
- `6_topic_modeling.ipynb`: Topic modeling using Bertopic.
- `7_tag_popularity_and_graph.ipynb`: Tag popularity and graph analysis.
- `8_rest_service_performance_metrics.ipynb`: Performance metrics for REST service

## 🐳 Docker Setup
You can also use Docker to containerize the application.

Build the Docker Image (from the root directory):
```bash
docker build -t tag-recommender:latest -f docker/Dockerfile .
```
Run the Docker Container
```bash
docker run --name tag-recommender -p 8000:8000 tag-recommender:latest
```
