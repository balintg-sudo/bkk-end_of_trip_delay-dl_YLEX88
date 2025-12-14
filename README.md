# Deep Learning Class (VITMMA19) Project Work

## Project Details

### Project Information

- **Selected Topic**: End-of-trip delay prediction
- **Student Name**: Bálint Gergely
- **Aiming for +1 Mark**: No

### Solution Description

The objective of this project is to predict the arrival of BKK vehicles at their final destination. My model incorporates spatial dependencies (the network structure of stops) by constructing a graph where nodes represent stops, edges represent direct connections between stops, and edge weights correspond to the scheduled travel times. Since traditional models often overlook network topology, this solution aims to apply a graph-based approach utilizing GTFS Real-time data.

**Configuration:**

I have created a configuration file (`src/config.py`) that allows for the fine-tuning of the following parameters:

* **DAYS_TO_TRAIN**: The number of days to be used for training data (out of the available 16).
* **LOW_MEMORY switch**: If set to `True`, the program limits loaded data to 4,000,000 rows, saving significant resources and time.
    * *IMPORTANT:* If `True`, the model will train on 5 days and test on 1 day.
* **EPOCHS**: The number of training epochs.
* **BATCH_SIZE**: The size of the batches.
* **LEARNING_RATE**: Allows adjustment of the learning rate.

**Workflow:**

1.  The program downloads the raw data from Google Drive.
2.  It performs data loading and cleaning, then splits the data into Training/Validation and Test sets.
3.  It trains and evaluates the Baseline Linear Regression model, followed by the GNN.
4.  As a final step, it evaluates the Regression, the Heuristic baseline estimator, and the GNN on unseen data and outputs the statistics.

**Baseline Models:**

* **Linear Regression**
* **Heuristic Estimator (Naive Baseline):** Takes the current delay and assumes it will be the final delay (i.e., it assumes the vehicle will neither accumulate further delay nor recover time).

**Evaluation:**

The models were compared based on the following metrics:
* **MAE** (Mean Absolute Error)
* **RMSE** (Root Mean Square Error)
* **MedAE** (Median Absolute Error)
* **Max Error**
* **Accuracy percentages** within specific thresholds (+/- 1, 3, and 5 minutes).

> **IMPORTANT:** When I refer to TEST data in the final step, it is strictly unseen data, separate from the train/test split used during model training.

### Extra Credit Justification

No

---

## Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

### 1. Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

#### Run

To run the solution, use the following command. 
* Linux:
    ```bash
    bash ./run.sh
    ```

* Windows:
    ```bash
    .\run.ps1
    ```
* If it doesn't work:
* Windows:
    ```bash
    New-Item -ItemType Directory -Force -Path "log" | Out-Null; docker run -v ${PWD}/data:/app/data -v ${PWD}/log:/app/log --memory="12g" --memory-swap="-1" dl-project > ".\log\run.log" 2>&1
    ```
* Linux:
    ```bash
    mkdir -p log && docker run -v "$(pwd)/data":/app/data -v "$(pwd)/log":/app/log --memory="12g" --memory-swap="-1" dl-project > log/run.log 2>&1
    ```

If somehow the download doesn't starts:
You must mount your local data directory to `/app/data` inside the container.
You can download everything form here: https://drive.google.com/drive/folders/1IbiDXxosOCT7EUgTW1VLhrIkD46nzfpi?usp=drive_link

---
### File Structure and Functions

[Update according to the final file structure.]

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `00-download.py`: Downloads the CSV from Google Drive
    - `01-data-preprocessing.py`: Scripts for loading, cleaning, and preprocessing the raw data.
    - `02-training.py`: The main script for defining the model and executing the training loop and evaluates it on test.
    - `03-evaluation.py`: Scripts for evaluating the trained model on unseen data and generating metrics.
    - `config.py`: Configuration file containing hyperparameters (e.g., epochs) and paths.
    - `utils.py`: Helper functions and utilities used across different scripts.

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `Calculate_delays.ipynb`: Notebook for calculating the delays and checks the validity. The raw data uploaded comes from this.
    - `Data_exploration.ipynb`: Notebook for Cleaning and exploring the data.
    - `Baseline.ipynb`: Notebook for the basline models
    - `GNN.ipynb`: Notebook for creating the GNN

- **`log/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
    - `run.sh`: Run file for Linux
    - `run.ps1`: Run file for Windows

