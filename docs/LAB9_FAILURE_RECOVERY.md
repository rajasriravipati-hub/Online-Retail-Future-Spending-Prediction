# Lab 9: Pipeline Failure Handling and Recovery

## Aim

To implement and validate failure-handling and recovery mechanisms for an ML pipeline by simulating common failures and testing recovery strategies.

## Objectives

* Detect missing data and invalid datatypes.
* Validate the input dataset schema.
* Handle preprocessing errors.
* Recover from missing model artifacts using a backup or model retraining.
* Generate a report documenting the recovery results.

## Dataset

The experiment used the processed Online Retail dataset containing 4,326 valid rows and seven columns, including six input features and the target variable `FutureSpending`.

## Failure Scenarios and Results

| S. No. | Failure Scenario       | Recovery Mechanism                                     | Result    |
| ------ | ---------------------- | ------------------------------------------------------ | --------- |
| 1      | Missing data           | Detect and remove rows containing missing values       | RECOVERED |
| 2      | Invalid datatype       | Validate and convert data to appropriate numeric types | RECOVERED |
| 3      | Schema mismatch        | Check required columns and raise a validation error    | RECOVERED |
| 4      | Preprocessing error    | Catch the exception and apply data-cleaning recovery   | RECOVERED |
| 5      | Missing model artifact | Attempt backup recovery or retrain the model           | RECOVERED |

## Implementation

The recovery logic is implemented in `src/recovery_pipeline/failure_recovery.py`. The experiment is executed through `pipelines/run_lab9.py`.

The pipeline uses validation checks, exception handling, preprocessing recovery, model backup handling, and retraining when a usable backup is unavailable.

## Result

All five simulated failure scenarios were reported as RECOVERED. The pipeline generated `reports/lab9_recovery_report.json` containing the recovery results.

## Conclusion

The experiment demonstrated that the ML pipeline can detect simulated failures and apply recovery mechanisms instead of terminating immediately. The successful recovery results validate the implemented handling paths for the tested scenarios. These results demonstrate recovery behavior for the simulated failures; they do not establish robustness against every possible production failure.
