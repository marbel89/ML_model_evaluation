# 1. Create a class that performs hyperparameter tuning for a random forest DONE
# 2. Generalize class to handle other ML models DONE

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from rulesbased_prediction import DataSet
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


class MLModel:
    """
    A class used to represent and tune a passed model using RandomizedSearchCV.

    Attributes
    ----------
    ml_model :
    What model is to be used
    parameters : dict
    Dictionary of hyperparameters for RandomizedSearchCV to search over.
    n_jobs : int
    Number of jobs to run in parallel. -1 means using all processors.

    scoring : str or callable
    The scoring metric to use for evaluating the performance of the model.
    n_iter : int
    Number of parameter settings that are sampled during the search.

    random_state : int or None
    Seed used by the random number generator for reproducibility.

    Methods
    -------
    tune(X_features, y):
        Performs hyperparameter tuning on the training data.
    predict(X_features):
        Makes predictions on the provided features using the tuned model.
    """

    def __init__(self,
                 ml_model,
                 parameters: dict,
                 n_jobs: int,
                 scoring: str,
                 n_iter: int,
                 random_state: int):
        self.ml_model = ml_model
        self.clf = None
        self.parameters = parameters
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.n_iter = n_iter
        self.random_state = random_state

    def tune(self, X_features, y):
        """
        Performs hyperparameter tuning on the training data.

        Parameters
        ----------
        X_features : array-like or sparse matrix, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values (class labels) as integers or strings.
        """

        self.clf = RandomizedSearchCV(
            self.ml_model,
            self.parameters,
            n_jobs=self.n_jobs,
            scoring=self.scoring,
            n_iter=self.n_iter,
            random_state=self.random_state)

        self.clf.fit(X_features, y)

    def predict(self, X_features):
        if self.clf is None:
            raise ValueError("The model has not been tuned yet. Please call the `tune` method first.")
        return self.clf.predict(X_features)

    def evaluate(self, X_features, y):
        """
        Evaluates the model performance on the given test data.

        Parameters
        ----------
        X_features : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples to evaluate.
        y : array-like, shape (n_samples,)
            The true class labels.

        Returns
        -------
        dict
            A dictionary containing evaluation metrics.
        """
        if self.clf is None:
            raise ValueError("The model has not been tuned yet. Please call the `tune` method first.")

        predictions = self.clf.predict(X_features)
        roc_auc = roc_auc_score(y, predictions)
        accuracy = accuracy_score(y, predictions)

        return {
            'roc_auc': roc_auc,
            'accuracy': accuracy
        }


customer_obj = DataSet(
    feature_list=["total_day_minutes",
                  "total_day_calls",
                  "number_customer_service_calls"],
    file_name="customer_churn_data.csv",
    label_col="churn",
    pos_category="yes"
)

# Hyperparameter search grid
forest_parameters = {"max_depth": range(2, 6),
                     "min_samples_leaf": range(5, 55, 5),
                     "min_samples_split": range(10, 110, 5),
                     "max_features": [2, 3],
                     "n_estimators": [50, 100, 150, 200]}

forest = MLModel(ml_model=RandomForestClassifier(),
                 parameters=forest_parameters,
                 n_jobs=4,
                 scoring="roc_auc",
                 n_iter=10,
                 random_state=0)

gbm_parameters = {"max_depth": range(2, 6),
                  "min_samples_leaf": range(5, 55, 5),
                  "min_samples_split": range(10, 110, 5),
                  "max_features": [2, 3],
                  "n_estimators": [50, 100, 150, 200],
                  "learning_rate": [0.1, 0.2, 0.3]}

gbm = MLModel(ml_model=GradientBoostingClassifier(),
              parameters=gbm_parameters,
              n_jobs=4,
              scoring="roc_auc",
              n_iter=10,
              random_state=0)

forest.tune(customer_obj.train_features,
            customer_obj.train_labels)
gbm.tune(customer_obj.train_features,
         customer_obj.train_labels)

best_model = forest.clf.best_estimator_
print("Best Model:", best_model)
print("Best Parameters:", best_model.get_params())

evaluation_results = forest.evaluate(customer_obj.test_features, customer_obj.test_labels)
print("Evaluation Results:", evaluation_results)
