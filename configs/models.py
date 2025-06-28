import random
from pyspark.ml.classification import LogisticRegression, GBTClassifier

# -------------- Default Training Values --------------
DEFAULT_TRAINING_VALUES = {
    'start_date': '2023-01-01', # Start date for total data available for training, validation, and testing
    'end_date': '2024-06-01', # End date for total data available for training, validation, and testing
    'oot': 3,  # Number of out-of-time validation periods (each period is one month)
}

# ------------------ Gradient-Boosted Tree (GBT) Classifier -------------------
def get_gbt_classifier():
    """
    Returns a GBTClassifier instance and a correctly structured hyperparameter tuning set.
    This is the standard Spark equivalent of XGBoost.
    """
    gbt_classifier = GBTClassifier(
        labelCol='label',
        featuresCol='features',
        seed=42
    )

    # A random grid of common GBT hyperparameters to tune.
    # Note: GBTClassifier uses camelCase for its parameters.
    random_grid = [
        {
            gbt_classifier.maxDepth: random.choice([3, 5, 7]),
            gbt_classifier.stepSize: random.choice([0.01, 0.05, 0.1]), # Learning rate
            gbt_classifier.maxIter: random.choice([50, 100, 200]), # Number of trees
            gbt_classifier.subsamplingRate: random.choice([0.7, 0.8, 0.9, 1.0]),
            # GBT's feature sampling is simpler than XGBoost's
            gbt_classifier.featureSubsetStrategy: random.choice(["auto", "all", "sqrt", "log2"])
        }
        for _ in range(20)  # 100 random combinations
    ]

    return gbt_classifier, random_grid

# ------------------ Logistic Regression -------------------
def get_log_reg_classifier():
    """
    Returns a LogisticRegression instance and hyper parameter tuning set.
    """
    lr_classifier = LogisticRegression(labelCol='label', featuresCol='features', maxIter=100)

    # Randomly sample combinations
    random_grid = [
        {
            lr_classifier.regParam: random.choice([0.01, 0.02, 0.05, 0.1, 0.2]),
            lr_classifier.elasticNetParam: random.choice([0.0, 0.5, 1.0]),
            lr_classifier.maxIter: random.choice([100, 200, 500]),
            lr_classifier.threshold: random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        }
        for _ in range(100)  # x random combinations
    ]

    return lr_classifier, random_grid
