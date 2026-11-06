# Online Retail Future Spending Prediction

A Machine Learning Regression Project using Python, Scikit-learn, and MLOps practices to predict future customer spending based on historical purchasing behavior.

## 1. Project Overview

This project is a Machine Learning-based system that predicts the future spending of customers using their historical purchasing behavior from the Online Retail dataset.

The project transforms transaction-level retail data into customer-level features and applies regression algorithms to predict the `FutureSpending` of customers.

The project covers the complete Machine Learning workflow, including data preprocessing, exploratory data analysis, feature engineering, model development, evaluation, experiment tracking, pipeline automation, and failure recovery.

## 2. Problem Statement

Online retail businesses generate large amounts of customer transaction data. Analyzing this data helps businesses understand purchasing behavior and estimate how much customers may spend in the future.

The objective is to develop a Machine Learning regression model that predicts future customer spending based on historical purchasing behavior.

The predictions can support:

- Customer value analysis
- Customer segmentation
- Marketing decisions
- Customer retention strategies
- Business planning
- Data-driven decision making

## 3. Project Objectives

- Understand and analyze Online Retail transaction data.
- Perform exploratory data analysis (EDA).
- Clean and preprocess the dataset.
- Handle missing values and duplicate records.
- Analyze and handle outliers where required.
- Perform customer-level feature engineering.
- Select meaningful features for prediction.
- Build and compare regression models.
- Evaluate model performance.
- Apply hyperparameter tuning where required.
- Select and save the final trained model.
- Implement an automated Machine Learning pipeline.
- Track experiments and model performance.
- Implement continuous integration and failure recovery workflows.

## 4. Dataset Description

**Dataset:** Online Retail Dataset

**Source:** Kaggle – Online Retail Dataset

The dataset contains transaction-level information about customer purchases.

| Column | Description |
|---|---|
| InvoiceNo | Unique invoice number |
| StockCode | Product code |
| Description | Product description |
| Quantity | Number of products purchased |
| InvoiceDate | Date and time of transaction |
| UnitPrice | Price per unit |
| CustomerID | Unique customer identifier |
| Country | Customer's country |

The original dataset contains approximately 541,909 transaction records and 8 attributes.

The transaction-level dataset is processed and transformed into customer-level information for Machine Learning.

## 5. Target Variable

The target variable is:

`FutureSpending`

The regression model predicts a numerical value representing future customer spending based on historical purchasing behavior.

## 6. Features Used

The final Machine Learning dataset contains six customer-level features.

| Feature | Description |
|---|---|
| TotalSpent | Total amount spent by the customer |
| NumberOfOrders | Total number of orders placed |
| TotalQuantity | Total quantity of products purchased |
| AverageOrderValue | Average value of each order |
| UniqueProducts | Number of unique products purchased |
| AverageUnitPrice | Average unit price of purchased products |

## 7. Technologies Used

### Programming Language
- Python

### Data Analysis and Visualization
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn
- Ridge Regression

### Model Persistence
- Joblib

### MLOps and Experiment Tracking
- MLflow
- Git
- GitHub Actions
- Pytest

### Development Environment
- Visual Studio Code
- Jupyter Notebook
- Python Virtual Environment

## 8. Machine Learning Workflow

The project follows a structured Machine Learning workflow:

```text
Dataset Understanding
        |
        v
Data Loading
        |
        v
Data Cleaning
        |
        v
Exploratory Data Analysis
        |
        v
Missing Value Handling
        |
        v
Duplicate Handling
        |
        v
Outlier Analysis
        |
        v
Feature Engineering
        |
        v
Feature Selection
        |
        v
Train-Test Split
        |
        v
Feature Scaling
        |
        v
Model Building
        |
        v
Model Evaluation
        |
        v
Hyperparameter Tuning
        |
        v
Final Model Selection
        |
        v
Model Saving
```

## 9. Model Development

The project uses regression algorithms to estimate future customer spending.

**Selected Model:** Ridge Regression

Ridge Regression is a regularized linear regression algorithm that uses L2 regularization to reduce overfitting and improve model generalization.

### Evaluation Metrics

The model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

### Model Performance

| Metric | Value |
|---|---:|
| MAE | 0.269659 |
| RMSE | 2.038925 |
| R² Score | 0.999996 |

*These are the previously recorded Lab 7 results. Metric interpretation depends on the target variable's scale and the evaluation split.*

## 10. MLOps Experiments

The project includes MLOps experiments covering automated Machine Learning workflows, experiment tracking, continuous integration, and failure recovery.

### Lab 7 – MLOps Pipeline Automation

- Automated model training and evaluation.
- Implemented pipeline execution.
- Integrated experiment tracking using MLflow.
- Recorded model performance metrics.
- Organized pipeline execution and artifacts.

Documentation: `docs/LAB7_PIPELINE.md`

### Lab 8 – Continuous Integration

- Configured GitHub Actions.
- Implemented automated validation workflows.
- Executed tests during code integration.
- Used automated checks to improve project reliability.

Documentation: `docs/LAB8_CI_REPORT.md`

### Lab 9 – Failure Handling and Recovery

Implemented and tested recovery mechanisms for:

- Missing data
- Invalid data types
- Schema mismatches
- Preprocessing errors
- Missing model artifacts

Documentation: `docs/LAB9_FAILURE_RECOVERY.md`

## 11. Project Structure

```text
Online_Retail_ML_project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── models/
│
├── src/
│   ├── monitoring/
│   └──
│
├── scripts/
│
├── tests/
│
├── docs/
│   ├── LAB7_PIPELINE.md
│   ├── LAB8_CI_REPORT.md
│   └── LAB9_FAILURE_RECOVERY.md
│
├── reports/
│
├── requirements.txt
├── README.md
└── .gitignore
```

*The structure above is a simplified overview. Adjust folder names to match the actual repository.*

## 12. How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/rajasriravipati-hub/Online-Retail-Future-Spending-Prediction.git
```

### Step 2: Navigate to the Project

```bash
cd Online-Retail-Future-Spending-Prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Open the Jupyter Notebook

```bash
jupyter notebook
```

Open the project notebook and execute the cells in sequence.

## 13. Expected Outcomes

- Understanding of retail customer purchasing behavior.
- Customer-level dataset preparation.
- Regression-based future spending predictions.
- Evaluation and comparison of Machine Learning models.
- Automated ML pipeline execution.
- MLflow experiment tracking.
- GitHub Actions-based continuous integration.
- Failure handling and recovery mechanisms.

## 14. Future Enhancements

- Develop an interactive prediction application.
- Integrate real-time customer transaction data.
- Improve model generalization using additional validation strategies.
- Explore advanced regression algorithms.
- Implement more comprehensive production monitoring.

## 15. Conclusion

The Online Retail Future Spending Prediction project demonstrates the application of Machine Learning regression techniques to retail transaction data. It covers data preparation, feature engineering, model development, evaluation, and MLOps practices through Lab 9.

The project provides practical experience in building reproducible Machine Learning workflows and managing model development using Python, Scikit-learn, MLflow, Git, and GitHub Actions.

## Author

**Raja Sri Ravipati**

B.Tech – Computer Science and Engineering (Artificial Intelligence and Machine Learning)

Vignan's Foundation for Science, Technology and Research

GitHub: [rajasriravipati-hub](https://github.com/rajasriravipati-hub)

**Project:** Online Retail Future Spending Prediction

**Project Type:** Supervised Machine Learning – Regression
