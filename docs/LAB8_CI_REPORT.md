# Lab 8: Continuous Integration for Online Retail MLOps

## 1. Objective

To configure a GitHub Actions continuous integration workflow for the Online Retail MLOps project.

## 2. Workflow File

`.github/workflows/lab8_ci.yml`

## 3. Triggers

The workflow is configured to run on:

* Pushes to the `main` and `lab8-ci` branches.
* Pull requests targeting the `main` branch.

## 4. CI Steps

1. Check out the repository.
2. Set up Python 3.11.
3. Install project dependencies from `requirements.txt`.
4. Check Python syntax in the source, pipeline, and test directories.
5. Verify that required Lab 7 JSON reports exist and contain valid JSON.

## 5. Expected Outcome

The workflow should complete successfully when the required dependencies install, Python syntax checks pass, and all required report files are present and valid.

## 6. Conclusion

The Lab 8 workflow defines automated repository checks using GitHub Actions. The workflow must be pushed to GitHub and executed successfully before CI success can be confirmed.
