# 🌸 Iris ML — Flower Classification with Docker

A containerized machine learning application that trains a **Random Forest classifier** on the classic Iris dataset and runs predictions with confidence scores — all inside a single Docker image.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Model Performance](#model-performance)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Sample Output](#sample-output)
- [Project Structure](#project-structure)

---

## Overview

This project packages a complete ML pipeline into a Docker container:

- Loads the Iris dataset (150 samples, 4 features)
- Trains an 80/20 stratified train/test split
- Fits a **RandomForestClassifier** with 100 trees
- Predicts flower species with per-class probabilities
- Serves results via a local web interface on port `5000`

**Target classes:** `setosa` · `versicolor` · `virginica`

---

## Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | **90.00%** |
| Macro Avg Precision | 0.90 |
| Macro Avg Recall | 0.90 |
| Macro Avg F1-Score | 0.90 |

**Per-class breakdown:**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| setosa | 1.00 | 1.00 | 1.00 |
| versicolor | 0.82 | 0.90 | 0.86 |
| virginica | 0.89 | 0.80 | 0.84 |

**Feature importances:**

```
petal width (cm)    0.4372  █████████████████
petal length (cm)   0.4315  █████████████████
sepal length (cm)   0.1163  ████
sepal width (cm)    0.0150
```

---

## Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Windows CMD / PowerShell (or any terminal on macOS/Linux)

---

## Quick Start

### 1. Install & Start Docker Desktop

Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) and open it. Wait until the status shows **Docker Desktop is running**.

### 2. Load the Docker Image

Open a terminal and run — replacing the path with wherever your `iris-ml.tar` file is:

```bash
docker load -i "C:\path\to\iris-ml.tar"
```

**Examples:**
```bash
docker load -i "C:\Users\YourName\Desktop\iris-ml.tar"
docker load -i "C:\Users\YourName\Downloads\iris-ml.tar"
```

**Expected output:**
```
Loaded image: iris-ml:latest
```

### 3. Verify the Image

```bash
docker images
```

You should see:

```
REPOSITORY   TAG      IMAGE ID       SIZE
iris-ml      latest   fd3f82e783b8   694MB
```

### 4. Run the Container

```bash
docker run -p 5000:5000 iris-ml
```

> If port `5000` is already in use, try port `8000`:
> ```bash
> docker run -p 8000:5000 iris-ml
> ```

### 5. Open the App

Visit in your browser:

```
http://localhost:5000
```

or `http://localhost:8000` if you used port 8000.

### 6. Stop the Container

Press `Ctrl + C` in the terminal window where the container is running.

---

## Usage

### Finding the `.tar` File Path (Windows)

| Method | How |
|---|---|
| File Explorer | Right-click file → Properties → Location |
| CMD drag-and-drop | Drag the file into CMD — path auto-fills |
| Desktop shortcut | Usually `C:\Users\YourName\Desktop\iris-ml.tar` |

---

## Sample Output

### Training

```
=======================================================
Iris Flower Classification — Training
=======================================================
Dataset loaded: 150 samples, 4 features
Classes: ['setosa', 'versicolor', 'virginica']
Split: 120 train | 30 test (80/20, stratified)
RandomForestClassifier trained (100 trees)
Test accuracy: 90.00%
Model saved to model.joblib
=======================================================
```

### Prediction

```
Sample : Classic setosa
Input  : sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2
▶ Predicted : SETOSA (confidence 100.0%)

Sample : Borderline sample
Input  : sepal_length=5.9, sepal_width=3.0, petal_length=4.8, petal_width=1.8
▶ Predicted : VERSICOLOR (confidence 53.0%)
   setosa     0.0%  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
   versicolor 53.0% ███████████████░░░░░░░░░░░░░░░
   virginica  47.0% ██████████████░░░░░░░░░░░░░░░░
```

---

## Project Structure

```
iris-ml/
├── data/
│   ├── train.csv        # 120-sample training split
│   └── test.csv         # 30-sample test split
├── model.joblib         # Saved RandomForest model
├── Dockerfile
└── README.md
```

---

## License

This project is for educational and demonstration purposes.
