# Online Retail Customer Future Spending Prediction

## 1. Project Overview

This project is a Machine Learning-based system that predicts the future spending of customers using their historical purchasing behavior from an Online Retail dataset.

The project converts transaction-level retail data into customer-level features and applies regression algorithms to predict the `FutureSpending` of customers.

The complete Machine Learning workflow is implemented in a Jupyter Notebook, and the final trained model is deployed through a Streamlit web application.

---

## 2. Problem Statement

Online retail businesses generate large amounts of customer transaction data. Analyzing this data can help businesses understand customer purchasing behavior and estimate how much customers may spend in the future.

The objective of this project is to develop a Machine Learning regression model that predicts future customer spending based on historical purchasing behavior.

The prediction can help businesses with:

- Customer value analysis
- Customer segmentation
- Marketing decisions
- Customer retention strategies
- Business planning
- Data-driven decision making

---

## 3. Objectives

The main objectives of this project are:

- Understand and analyze Online Retail transaction data.
- Perform exploratory data analysis.
- Clean and preprocess the dataset.
- Handle missing values and duplicate records.
- Analyze and handle outliers where required.
- Perform customer-level feature engineering.
- Select meaningful features for prediction.
- Build multiple regression models.
- Evaluate and compare model performance.
- Apply hyperparameter tuning where required.
- Select the best-performing model.
- Save the final trained model.
- Develop a Streamlit application for prediction.
- Provide a simple interface for predicting future customer spending.

---

## 4. Dataset

The project uses the Online Retail dataset.

The dataset contains transaction-level information about customer purchases.

### Important Dataset Attributes

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

The original transaction-level dataset was processed and transformed into customer-level information for Machine Learning.

---

## 5. Target Variable

The target variable used in this project is:

`FutureSpending`

The objective of the model is to predict future customer spending using historical customer purchasing behavior.

---

## 6. Features Used

The final Machine Learning model uses the following six features:

| Feature | Description |
|---|---|
| TotalSpent | Total amount spent by the customer |
| NumberOfOrders | Total number of orders placed by the customer |
| TotalQuantity | Total quantity of products purchased |
| AverageOrderValue | Average value of each order |
| UniqueProducts | Number of unique products purchased |
| AverageUnitPrice | Average unit price of purchased products |

---

## 7. Technologies Used

### Programming Language

- Python

### Machine Learning and Data Science Libraries

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

### Model Saving

- Joblib

### Deployment

- Streamlit

### Development Tools

- Visual Studio Code
- Jupyter Notebook
- Python Virtual Environment

---

## 8. Machine Learning Workflow

The project follows a complete Machine Learning workflow:

```text
Dataset Understanding
        ↓
Data Loading
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Missing Value Handling
        ↓
Duplicate Handling
        ↓
Outlier Analysis
        ↓
Feature Engineering
        ↓
Feature Selection
        ↓
Train-Test Split
        ↓
Feature Scaling
        ↓
Model Building
        ↓
Model Evaluation
        ↓
Hyperparameter Tuning
        ↓
Final Model Selection
        ↓
Model Saving
        ↓
Streamlit Deployment

