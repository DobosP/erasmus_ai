# Machine Learning in Practice

## score_solution documentation

Using the script score_solution.py you can test
your solution using the train data.

The script takes 3 arguments. or you can *hardcode* them in the script. If you harcode them comment the args parses lines of code.


-  submission:  Relative/absolute path to you submision csv file
-  train_data:  Relative/absolute path to you train data csv file
-  percent:     Percent of train data for testing data

OBS!!

We use the last "percent" percent from data for training.

The test will be done using the test data in interval [truncate_point, size_data_train] where truncate_point is computed with the next formula.

truncate_point = int(size_data_train*(float(100 - percent)/100))

Ex.

If percent is 10. First 90% of date is use to train the model. The rest 10% is used for training.




Usage:
`python score_solution.py sample_submission.csv train.csv 10`

Where:
-  sample_submission.csv: The relative path to the solution file.
- train.csv: The relative path to the train data.
- 10: the percent
