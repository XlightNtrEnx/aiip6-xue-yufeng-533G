# AIIP 6 Technical Assessment

## a. Full name and email address

Xue Yufeng yufeng_xue@mymail.sutd.edu.sg

## b. Overview of the submitted folder and structure

### data

Contains bmarket.db

### src

main.py puts the data (./data.py) through the pipelines (./pipelines).

In the pipelines folder, we have multiple pipelines put into their .py files.

lr.py - Logistic Regression

nb.py - Naive Bayes

nn.py - Neural Network

svm.py - Support Vector Machine

tree.py - Tree

```py
# Example pipeline in lr.py
pipes = [] # Exported for main.py to retrieve and execute

pipes.append(Pipeline(steps=[
    standard_preprocessor(), # Function will be explained below
    ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced"))
]))
```

data.py processes the dataframe and exports the features as X (DataFrame) and target as y (Series)

preprocessor.py contains function for constructing a preprocessor in a pipeline. standard_preprocessor will return a standard one. There is many more functions for customizing a preprocessor if necessary.

```py
def standard_preprocessor():
    return ("preprocessor", ColumnTransformer(
        transformers=[
            ("boolean", Pipeline(steps=[
                ("to_categorical", FunctionTransformer(lambda X: X.astype('object'))),
                ("onehot", OneHotEncoder(handle_unknown="error", sparse_output=False))
            ]), bool_features),
            ("categorical", Pipeline(steps=[
                ("imputation_constant", SimpleImputer(fill_value="missing", strategy="constant")),
                ("onehot", OneHotEncoder(handle_unknown="error", sparse_output=False))
            ]), cat_features)
            ("numerical", Pipeline(steps=[
                ("imputation_constant", SimpleImputer(fill_value=-1, strategy="constant")),
                ("scaler", StandardScaler()),
            ]), num_features)

        ]
    ))
```

logger.py configures logging and exports a logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

## c. Instructions for executing the pipeline and modifying any parameters

```py
def evaluate(pipe: Pipeline, logger: logging.Logger):
    '''
    This function feeds the data to the pipeline and evaluates it using the output.

    Parameters:
    pipe (Pipeline): The pipeline
    logger (logging.Logger): The logger used to log pipeline structure and evaluation metrices

    Returns:
    float, float: The tuple of recall and precision of the pipe
    '''
    pass

top = []
def run_evaluate(pipelines: Pipeline[], filename: str):
    '''
    This function calls evaluate on each pipe with a logger to log to the specified file with the filename
    After every evaluate, it appends top with (score, precision, recall, idx, filename).

    Parameters:
    pipelines (Pipeline[]): The pipelines.
    filename (str): The log file for the evaluation metrices of these pipelines
    '''
    pass

run_evaluate(lr_pipes, "lr") # Second arg is the name of the log file that will be created for the pipe
run_evaluate(tree_pipes, "tree")
run_evaluate(svm_pipes, "svm")
run_evaluate(nb_pipes, "nb")
run_evaluate(nn_pipes, "nn")

top_logger.info(f"\nTop 3:\n{sorted(top)[-3:]}")
```

## d. Description of logical steps/flow of the pipeline

1. data.py exports X and y
2. main.py takes these and feeds into all pipelines
3. each pipeline consists of a preprocessor that processes the data and feeds into the model
4. the pipeline internally trains the model repeatedly based on the data
5. pipeline is then tested against a test portion of a y and evaluated

## e. Overview of key findings from the EDA conducted in Task 1 and the choices made in the pipeline based on these findings

The data is heavily imbalanced in the target column (88% unsubscribed). Certain pipelines were configured to account for this.

Invalid data was present. They were identified by the Age column where there were people older than 122 which is impossible as that's the oldest age reached.

```py
processed_df = processed_df[(processed_df['Age (years)'] >= 0) & (processed_df['Age (years)'] <= 122)]
```

There are also useless features (poor discriminatory power) that were dropped.

```py
processed_df.drop('Housing Loan', axis=1, inplace=True)
processed_df.drop('Personal Loan', axis=1, inplace=True)
```

Campaign Calls feature had illogical values (-ve values) so they were converted to +ve in the reason that they might have been mishandled (attaching minus sign by accident).

```py
processed_df["Campaign Calls"] = processed_df["Campaign Calls"].apply(lambda x: abs(x))
```

Features with high range and skew were logged to enable better convergence for the models.

```py
processed_df['Age (years)'] = np.log1p(processed_df['Age (years)'])
processed_df['Campaign Calls'] = np.log1p(processed_df['Campaign Calls'])
```

## f. Describe how the features in the dataset are processed (summarised in a table)

All values 'unknown' were replaced with NA

| Feature               | Action                                                             |
| --------------------- | ------------------------------------------------------------------ |
| Age                   | converted to str, logged values, renamed to Age (years)\_log       |
| Campaign Calls        | converted -ve to +ve, logged values, renamed to Campaign Calls_log |
| Credit Default        | converted to bool                                                  |
| Client ID             | dropped                                                            |
| Contact Method        | converted to lowercase, merged similar values                      |
| Education Level       | -                                                                  |
| Housing Loan          | dropped                                                            |
| Marital Status        | -                                                                  |
| Personal Loan         | dropped                                                            |
| Previous Contact Days | converted 999 to NaN                                               |
| Subscription Status   | converted to bool, renamed to target                               |

## g. Explanation of your choice of models for each machine learning task.

I used every model I could think of since we can never be sure which model is good, but I'll explain the rationale for each configuration.

Logistic Regression - Set class_weight="balanced" since we're dealing with imbalanced data

Naive Bayes - GaussianNB because of a continuous numerical data.

Neural Network - SGD because it seemed to perform better than Adam and deeper layers couldn't work because of vanishing gradient problem.

Support Vector Machine - Set class_weight="balanced", used random undersampling and SMOTE since we're dealing with imbalanced data

Tree - Experimented with different scale_pos_weight because of imbalanced data. Found a formula that works and applied it to get that value.

## h. Evaluation of the models developed. Any metrics used in the evaluation should also be explained.

The precision and recall of True predictions are valuable, because they affect the profits of the company using the predictions, assuming that the company allocates resources to those predicted True.

If the company directs resources to all who were predicted True (predicted that they will be subscribed), the precision becomes an indicator of the efficiency of those resources as it measures how many of those predicted to be True are actually subscribed, and indicates the likely percentage of those resources succeeding. For recall in this case, it's an indicator of how many who are actually True will be predicted to be true, and shows the non-missed profits of these resources.

The other metrices are useless.

To determine the best model, I use the formula below.

```py
magnitude = (recall + precision*1.5) / 2
imbalance = abs(recall - precision)
balance = 1 - imbalance
score = magnitude * balance
```

I multiply precision by 1.5 because I think it is more important than recall, since it correlates to efficiency.

I get the top 3 performers here.

```py
# (score, precision, recall, idx, filename)
[(0.33404159738691414, 0.6380597014925373, 0.21033210332103322, 1, 'nn'), (0.38477105440977505, 0.5382653061224489, 0.25953259532595324, 2, 'tree'), (0.42073566619329855, 0.3728813559322034, 0.3247232472324723, 1, 'tree')]
```

3rd place: 2nd neural network in nn.py

2nd place: 3rd tree in tree.py

1st place: 2nd tree in tree.py

```py
# 3rd place
pipes.append(Pipeline(steps=[
    standard_preprocessor(),
    make_classifier(MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu", solver="sgd", max_iter=1500))
]))

# 2nd place
pipes.append(make_pipeline(
    standard_preprocessor()[1],
    random_under_sampler()[1],
    XGBClassifier(eval_metric='aucpr', scale_pos_weight=0.136425648)
))

# 1st place
pipes.append(make_pipeline(
    standard_preprocessor()[1],
    smote()[1],
    XGBClassifier(eval_metric='aucpr', scale_pos_weight=0.9)
))
```

## i. Other considerations for deploying the models developed

NA
