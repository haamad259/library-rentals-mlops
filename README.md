# Library Rentals MLOps Project

This repository contains an end-to-end Machine Learning pipeline to predict library book rentals.

## Project Files

* **Library_Rentals_Hail_Project.ipynb**: Jupyter Notebook containing data analysis, model training, and evaluation.
* **model.pkl**: Trained Machine Learning model.
* **scaler.pkl**: Feature scaler used for preprocessing.
* **app.py**: FastAPI application serving predictions.
* **Dockerfile**: Configuration file to containerize the application.
* **requirements.txt**: Python dependencies required for the project.

## How to Run Locally

### 1. Install dependencies:
`ash
pip install -r requirements.txt

2. Run the FastAPI Application:

uvicorn app:app --reload


Docker Instructions
To run the project inside a Docker container:

1. Build the Docker Image:

docker build -t library-rentals .


2. Run the Docker Container:

docker run -p 8000:8000 library-rentals


