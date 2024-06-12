import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics


class DataSet:
    """
    A class used to represent a dataset for customer churn analysis.

    Attributes
    ----------
    customer_data : pd.DataFrame
        The entire dataset loaded from the CSV file.
    train_data : pd.DataFrame
        The training subset of the data.
    test_data : pd.DataFrame
        The testing subset of the data.
    feature_list : list
        A list of feature column names.
    train_features : pd.DataFrame
        Features from the training data.
    test_features : pd.DataFrame
        Features from the testing data.
    train_labels : pd.Series
        Binary labels for the training data.
    test_labels : pd.Series
        Binary labels for the testing data.

    Methods
    -------
    get_summary_plots():
        Displays histograms of each feature in the feature list.
    get_model_metrics(train_pred, test_pred):
        Prints precision scores for the training and testing predictions.
    """

    def __init__(self, feature_list: list, file_name: str, label_col: str, pos_category: str):
        """
        Constructs all the necessary attributes for the DataSet object.

        Parameters
        ----------
        feature_list : list
            List of feature column names to be used in the model.
        file_name : str
            Path to the CSV file containing the dataset.
        label_col : str
            Name of the column containing the labels.
        pos_category : str
            Category value that represents a positive label in the label column.
        """
        # Load the dataset from the CSV file
        self.customer_data = pd.read_csv(file_name)

        # Print the column names of the dataset for verification
        # print(self.customer_data.columns)

        # Split the data into training and testing sets (70% train, 30% test)
        self.train_data, self.test_data = train_test_split(
            self.customer_data,
            train_size=0.7,
            random_state=0
        )

        # Reset the index of the training data
        self.train_data = self.train_data.reset_index(drop=True)

        # Reset the index of the testing data
        self.test_data = self.test_data.reset_index(drop=True)

        # Store the feature list
        self.feature_list = feature_list

        # Extract the features from the training data
        self.train_features = self.train_data[feature_list]

        # Extract the features from the testing data
        self.test_features = self.test_data[feature_list]

        # Create binary labels for the training data (1 if 'yes', else 0)
        self.train_labels = self.train_data[label_col].map(lambda key: 1 if key == pos_category else 0)

        # Create binary labels for the testing data (1 if 'yes', else 0)
        self.test_labels = self.test_data[label_col].map(lambda key: 1 if key == pos_category else 0)

    def get_summary_plots(self):
        """Displays histograms of each feature in the feature list."""
        for feature in self.feature_list:
            self.train_data[feature].hist()
            plt.title(feature)
            plt.show()

    def get_model_metrics(self, train_pred: pd.Series, test_pred: pd.Series):
        """
        Prints precision scores for the training and testing predictions.

        Parameters
        ----------
        train_pred : pd.Series
            Predicted labels for the training data.
        test_pred : pd.Series
            Predicted labels for the testing data.
        """
        print("Rules based prediction train precision = ", metrics.precision_score(self.train_labels, train_pred))
        print("Rules based prediction test precision = ", metrics.precision_score(self.test_labels, test_pred))


# Initialize the DataSet object
customer_obj = DataSet(
    feature_list=["total_day_minutes", "total_day_calls", "number_customer_service_calls"],
    file_name="customer_churn_data.csv",
    label_col="churn",
    pos_category="yes"
)

# Optional: Generate summary plots for the features
# customer_obj.get_summary_plots()

# Add a binary feature indicating high service calls to the training data
customer_obj.train_data["high_service_calls"] = customer_obj.train_data.number_customer_service_calls.map(lambda val: 1 if val > 3 else 0)

# Add a binary feature indicating the presence of an international plan to the training data
customer_obj.train_data["has_international_plan"] = customer_obj.train_data.international_plan.map(lambda val: 1 if val == "yes" else 0)

# Create a rules-based prediction for the training data
customer_obj.train_data["rules_pred"] = [max(calls, plan) for calls, plan in zip(customer_obj.train_data.high_service_calls, customer_obj.train_data.has_international_plan)]

# Add a binary feature indicating high service calls to the testing data
customer_obj.test_data["high_service_calls"] = customer_obj.test_data.number_customer_service_calls.map(lambda val: 1 if val > 3 else 0)

# Add a binary feature indicating the presence of an international plan to the testing data
customer_obj.test_data["has_international_plan"] = customer_obj.test_data.international_plan.map(lambda val: 1 if val == "yes" else 0)

# Create a rules-based prediction for the testing data
customer_obj.test_data["rules_pred"] = [max(calls, plan) for calls, plan in zip(customer_obj.test_data.high_service_calls, customer_obj.test_data.has_international_plan)]

# Evaluate the model metrics using the rules-based predictions
customer_obj.get_model_metrics(customer_obj.train_data.rules_pred, customer_obj.test_data.rules_pred)
