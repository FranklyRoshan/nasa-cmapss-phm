# NASA C-MAPSS: Remaining Useful Life (RUL) Prediction

An end-to-end machine learning and deep learning project for estimating the **Remaining Useful Life (RUL)** of turbofan engines using the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset.

The notebook (`main.ipynb`) implements a leakage-aware workflow covering data ingestion, domain-specific ETL, exploratory data analysis, preprocessing, time-series feature engineering, model training, validation-based model selection, final training, and test-set evaluation.

## Project Objectives

- Estimate the number of operational cycles remaining before an engine reaches its end of life.
- Explore sensor behavior and degradation patterns under different operating conditions.
- Build and compare traditional machine learning and deep learning regression models.
- Prevent data leakage through engine-level splitting and training-only preprocessing.
- Evaluate the selected model using RMSE, MAE, and the NASA asymmetric scoring function.
- Save reusable preprocessing and model artifacts for inference.

## Dataset

The project supports all four standard C-MAPSS subsets:

| Subset | Operating conditions | Fault modes |
|---|---:|---:|
| FD001 | 1 | 1 |
| FD002 | 6 | 1 |
| FD003 | 1 | 2 |
| FD004 | 6 | 2 |

Each subset uses the standard training, test, and test RUL files:

- `train_FD00X.txt`
- `test_FD00X.txt`
- `RUL_FD00X.txt`

The data contains engine identifiers, cycle numbers, operating settings, and sensor measurements. Training engines run to failure; test engines are observed up to an earlier point, with the corresponding remaining-life labels supplied separately.

**Dataset access:** Obtain the NASA C-MAPSS dataset from an authorized source and configure one of the supported input methods below. The notebook does not bundle the dataset.

## Models Implemented

The notebook trains and compares six regression approaches:

| Model | Approach |
|---|---|
| Random Forest | Bagged decision-tree regression using engineered window-level features |
| XGBoost | Gradient-boosted tree regression using engineered window-level features |
| LSTM | Recurrent neural network for sequential sensor data |
| CNN-LSTM | 1D convolutional feature extraction followed by LSTM sequence modeling |
| Temporal Convolutional Network (TCN) | Dilated causal convolutions with residual connections |
| Temporal Transformer | Self-attention with positional encoding for temporal relationships |

The best model is selected using the validation-tail RMSE. The official test labels are reserved for final evaluation and are not used for hyperparameter tuning or architecture selection.

## Workflow

The notebook is organized into 18 sections:

1. **Environment Setup and Imports** — install and import dependencies, configure the run, verify the environment, and set random seeds.
2. **Data Ingestion** — load the selected subset's training data, test data, and test RUL labels.
3. **Pre-EDA Domain-Specific ETL** — profile, clean, and validate the raw data.
4. **Engine-Level Split, Imputation, and RUL Labels** — split training engines into training and validation partitions, fit imputation on training data, and construct RUL targets.
5. **Exploratory Data Analysis** — inspect sensor distributions, operating conditions, variance, and outliers.
6. **Leakage-Safe Preprocessing** — select sensors and fit operating-condition-based normalization using the training partition only.
7. **EDA on Normalized Training Data** — examine relationships, correlations, sensor degradation, and piecewise RUL behavior.
8. **Feature Engineering for Traditional ML** — generate per-engine sliding windows and extract window-level features.
9. **Random Forest Regression** — train, evaluate, tune, and persist a Random Forest model.
10. **XGBoost Regression** — train, evaluate, tune, and persist an XGBoost model.
11. **Feature Engineering for Deep Learning** — prepare sequence windows, align targets, pad or truncate sequences, and create tensors and data loaders.
12. **LSTM Regression** — train and evaluate an LSTM model.
13. **Hybrid CNN-LSTM Regression** — train and evaluate a CNN-LSTM model.
14. **Temporal Convolutional Network** — implement and evaluate a causal, dilated TCN.
15. **Temporal Transformer Regression** — implement and evaluate a Transformer-based regressor.
16. **Model Comparison and Selection** — compare all six models on the validation partition and select the model using validation-tail RMSE.
17. **Final Training and Test Evaluation** — train the selected model using the full training data, generate test predictions, and calculate final metrics.
18. **Final Project Verification** — run checks for dataset separation, engine boundaries, temporal leakage, prediction alignment, and saved artifacts.

## Leakage Prevention

The workflow is designed to avoid using future or held-out information during model development:

- The training/validation split is performed at the **engine level**, so an engine does not appear in both partitions.
- The imputer is fitted on the training partition and then applied to validation and test data.
- Operating-condition normalization components are fitted on training data only and reused without refitting.
- Sliding windows are constructed within individual engine histories; windows do not cross engine boundaries.
- Validation engines are held out from model fitting.
- Hyperparameter tuning and model selection use validation results, not the official test labels.
- Test RUL labels are used only for final evaluation.

## Evaluation Metrics

The notebook reports:

- **RMSE (Root Mean Squared Error):** penalizes larger prediction errors more strongly.
- **MAE (Mean Absolute Error):** average absolute prediction error.
- **NASA score:** asymmetric penalty that assigns a higher cost to late RUL predictions than to early predictions.
- **Late predictions (%):** percentage of predictions for which predicted RUL exceeds the true RUL.

For the NASA score, the prediction error is defined as:
$$
d = \widehat{\mathrm{RUL}} - \mathrm{RUL}
$$

$$
s(d) = \begin{cases} e^{-d/13} - 1, & d < 0 \\ e^{d/10} - 1, & d \geq 0 \end{cases}
$$


The total NASA score is the sum of the individual penalties. A lower score is better.

## Requirements

The notebook uses Python and the following main packages:

- NumPy
- pandas
- scikit-learn
- SciPy
- PyTorch
- XGBoost
- Plotly
- joblib
- safetensors
- kagglehub
- python-dotenv
- SQLAlchemy
- PyMySQL

Install the dependencies in your Python environment:

```bash
pip install numpy pandas scikit-learn scipy torch xgboost plotly joblib safetensors kagglehub python-dotenv sqlalchemy pymysql nbformat
```

The notebook also contains a package-installation cell. After installing packages in Jupyter, restart the kernel if required.

## Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If the repository does not contain a `requirements.txt` file, use the installation command in the [Requirements](#requirements) section.

### 3. Configure the notebook

Open `main.ipynb` and go to **Section 1.3 — Project Configuration**.

Set the dataset subset and data source:

```python
DATASET_ID = "FD002"       # FD001 | FD002 | FD003 | FD004
DATA_SOURCE = "kaggle_api" # local | kaggle_api | mysql
ARTIFACT_DIR = os.getenv("CMAPSS_OUT_DIR", "artifacts")
```

The notebook's initial configuration uses `FD002` and `kaggle_api`.

### 4. Configure data access

Choose one of the following input methods.

#### Option A: Local dataset files

Set:

```python
DATA_SOURCE = "local"
```

Configure the dataset directory in a `.env` file:

```dotenv
CMAPSS_DATA_DIR=/path/to/cmapss/data
```

The notebook searches below this directory for the selected subset's `train`, `test`, and `RUL` text files.

#### Option B: Kaggle API through kagglehub

Set:

```python
DATA_SOURCE = "kaggle_api"
```

Add the Kaggle dataset handle to `.env`:

```dotenv
KAGGLE_DATASET_HANDLE=your-kaggle-username/your-dataset-slug
```

Configure the Kaggle credentials required by your Kaggle environment before running the notebook. The dataset handle must point to a dataset containing the expected C-MAPSS text files.

#### Option C: MySQL

Set:

```python
DATA_SOURCE = "mysql"
```

Configure the database connection in `.env`:

```dotenv
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=your_database
MYSQL_USER=your_username
MYSQL_PASSWORD=your_password
MYSQL_TABLE=your_table
```

The table is expected to contain the standard raw columns plus `dataset_id` and `split` fields. The notebook reads the `train` and `test` splits for the selected dataset. The test RUL labels must also be available in the format expected by the notebook's MySQL ingestion code.

> Keep credentials in `.env` or another secure configuration mechanism. Do not commit passwords, API tokens, or private credentials to version control.

### 5. Run the notebook

Launch Jupyter:

```bash
jupyter lab
```

Open `main.ipynb` and execute the cells **from top to bottom**. The notebook is stateful, so running sections out of order may cause missing-variable or inconsistent-state errors.

## Configuration

The main experiment settings are centralized in Section 1.3.

| Parameter | Initial value | Description |
|---|---:|---|
| `DATASET_ID` | `FD002` | C-MAPSS subset |
| `DATA_SOURCE` | `kaggle_api` | Data ingestion method |
| `RUL_CAP` | `125` | Upper cap for piecewise-linear training RUL targets |
| `WINDOW_SIZE` | `30` | Sliding-window length in cycles |
| `VALIDATION_SIZE` | `0.2` | Fraction of training engines held out for validation |
| `RANDOM_STATE` | `42` | Random seed |
| `BATCH_SIZE` | `128` | Neural-network batch size |
| `LEARNING_RATE` | `1e-3` | Neural-network learning rate |
| `EPOCHS` | `30` | Main neural-network training epochs |
| `EARLY_STOPPING_PATIENCE` | `6` | Early-stopping patience |
| `DEVICE` | `cpu` | Requested compute device; CUDA is used only when available |

Additional preprocessing and hyperparameter-search settings are defined in the same configuration cell. Review them before running experiments.

## Output Artifacts

The notebook writes generated model and preprocessing files to `ARTIFACT_DIR`, which defaults to `artifacts/`. The output filenames include the selected dataset ID in lowercase.

| Artifact | Format | Purpose |
|---|---|---|
| `preprocessor_fd00x.joblib` | Joblib | Imputer, operating-condition components, scalers, selected sensors, and preprocessing configuration |
| `random_forest_fd00x.joblib` | Joblib | Random Forest model bundle and related metadata |
| `xgboost_fd00x.json` | JSON | XGBoost model |
| `xgboost_fd00x_meta.joblib` | Joblib | XGBoost feature and preprocessing metadata |
| `lstm_fd00x.safetensors` | Safetensors | LSTM model weights and metadata |
| `cnn_lstm_fd00x.safetensors` | Safetensors | CNN-LSTM model weights and metadata |
| `tcn_fd00x.safetensors` | Safetensors | TCN model weights and metadata |
| `temporal_transformer_fd00x.safetensors` | Safetensors | Transformer model weights and metadata |

`fd00x` represents the selected subset, such as `fd002`. The final selected model is also persisted by the final-training section using the selected model's corresponding artifact format.

To change the output directory, set `CMAPSS_OUT_DIR` in `.env`:

```dotenv
CMAPSS_OUT_DIR=artifacts
```

## Repository Structure

A typical layout is:

```text
.
├── main.ipynb
├── README.md
├── .env                 # Local configuration; do not commit
├── .gitignore
└── artifacts/           # Generated preprocessing and model artifacts
```

The dataset directory can be stored separately from the repository and referenced through `CMAPSS_DATA_DIR`.

## Reproducibility Notes

- The notebook sets seeds for Python, NumPy, and PyTorch.
- The initial random state is `42`.
- `QUICK_RUN` is available to reduce training epochs and search sizes for a faster smoke test.
- Results can vary with hardware, library versions, and nondeterministic operations.
- For a full experiment, review the configuration and disable quick-run settings as appropriate.

## Important Notes

- This is a regression workflow for estimating RUL from run-to-failure engine histories.
- The notebook supports FD001, FD002, FD003, and FD004, but each subset should be run and evaluated separately.
- The final reported test metrics depend on the selected dataset, configuration, random state, and installed package versions.
- No model performance values are hard-coded in this README; use the notebook's actual output for reported results.

## Future Improvements

Potential extensions include:

- Experiment tracking and systematic comparison across repeated random seeds.
- Additional feature-selection and feature-engineering strategies.
- Further model architectures and ensemble approaches.
- Model explainability and sensor-contribution analysis.
- A separate inference interface that loads the persisted preprocessing and model artifacts.

## References

- Saxena, A., Goebel, K., Simon, D., and Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. International Conference on Prognostics and Health Management.
- NASA Ames Prognostics Data Repository — C-MAPSS turbofan engine degradation simulation data.

---

**Project focus:** Remaining Useful Life estimation | Predictive Maintenance | Time-Series Regression | Machine Learning | Deep Learning
