# Lab 7: Online Retail MLOps Pipeline

## 1. Objective

To develop an end-to-end MLOps pipeline for the Online Retail Future Spending Prediction project. The pipeline includes data loading, validation, preprocessing, model training, evaluation, quality checking, model lifecycle management, inference, and failure-injection testing.

## 2. Dataset

* Raw dataset: `data/OnlineRetail.csv`
* Raw records: 541,909
* Processed dataset: `data/processed/lab5_processed_online_retail.csv`
* Processed records: 4,326
* Processed columns: 7

## 3. Pipeline Stages

1. Load the raw dataset.
2. Validate the raw dataset.
3. Preprocess the data.
4. Validate the processed output.
5. Train a Ridge Regression model and record the run in MLflow.
6. Evaluate the trained model.
7. Apply model quality gates.
8. Verify model lifecycle eligibility.
9. Generate inference predictions.
10. Perform failure-injection tests.

## 4. Model Details

* Algorithm: Ridge Regression
* Feature scaling: StandardScaler
* Train-test split: 80:20
* Random state: 42
* Ridge alpha: 1.0

## 5. Evaluation Results

* Training records: 3,460
* Testing records: 866
* MAE: 0.269659
* RMSE: 2.038925
* R²: 0.999996

These metrics are based on the current project's constructed target, `FutureSpending = TotalSpent * 0.10`. Therefore, the results should not be interpreted as proof of real-world future spending forecasting accuracy.

## 6. Quality Gate

The model passed the configured thresholds:

* MAE ≤ 1.0
* RMSE ≤ 5.0
* R² ≥ 0.90

These thresholds are project-specific settings.

## 7. Model Lifecycle

The model and scaler artifacts were verified. The lifecycle status was recorded as `eligible_for_promotion`. The existing production model was not changed.

## 8. Inference

Predictions were generated for 4,326 processed records and saved to `outputs/inference_predictions.csv`.

## 9. Failure-Injection Testing

The pipeline tested three scenarios:

* Missing required column
* Missing value
* Negative value

All three scenarios were detected successfully. The original dataset remained unchanged.

## 10. Important Output Files

* `artifacts/data_load_report.json`
* `artifacts/data_validation_report.json`
* `artifacts/preprocessing_summary_report.json`
* `artifacts/processed_output_validation_report.json`
* `artifacts/lab7_training_report.json`
* `reports/evaluation_report.json`
* `reports/quality_gate_report.json`
* `artifacts/production_model_report.json`
* `artifacts/inference_report.json`
* `artifacts/failure_simulation_report.json`
* `outputs/inference_predictions.csv`

## 11. Conclusion

The Lab 7 pipeline completed successfully. It automates the major data and model workflow stages, records evaluation results, checks configured quality thresholds, generates predictions, and tests detection of selected data failures.
