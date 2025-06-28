# CS611 Machine Learning Engineering Assignment 2

In this assignment, a complete ML workflow has been built with Airflow. The project is found in the respective GitHub repo for the CS611 course under: [GitHub-Repo](https://github.com/nicholas-voigt/SMU_CS611_Machine_Learning_Engineering/tree/main)


## Usage

### UI

1. build and run the docker container. enables webserver access on 0.0.0.0:8080
2. log in to airflow with user: admin pw: admin
3. enable both pipelines, the data and the ml pipeline. Workflow should start automatically. If not, clear history under admin, DAG runs.


### Terminal

1. build and run the docker container
2. connect to docker container via container id of the airflow_webserver container
3. something


## Structure

- /dags: contains the dag files with the execution logic for airflow (data pipeline and ml pipeline)
- /data: contains source data files in csv format. mounted to webserver and scheduler
- /scripts: contains the scripts which are executed from the airflow dag.
    - /scripts/data_processing: main processing logic for the data pipeline
    - /scripts/ml_processing: main processing logic for the ml pipeline
    - /scripts/configurations.py: general configurations for the file storage

