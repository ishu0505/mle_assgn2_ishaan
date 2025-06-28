# import os
# import argparse
# import tqdm
# from datetime import datetime
# import json

# from pyspark.sql import SparkSession
# from pyspark.ml import Pipeline
# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.evaluation import BinaryClassificationEvaluator

# from configs.data import training_data_dir, model_store_dir
# # from configs.models import get_gbt_classifier, get_log_reg_classifier
# from configs.models import get_gbt_classifier, get_log_reg_classifier


# if __name__ == "__main__":
#     # Parse command line arguments to define model type
#     parser = argparse.ArgumentParser(description='Train a machine learning model.')
#     parser.add_argument('--model_type', type=str, choices=['gbt', 'logreg'], required=True, help='Type of model to train: gbt for Gradient Boosted Trees or logreg for Logistic Regression.')
#     args = parser.parse_args()
#     model_type = args.model_type

#     # Initialize Spark session
#     spark = SparkSession.builder \
#         .appName("Model Trainer") \
#         .master("local[*]") \
#         .getOrCreate()
#     spark.sparkContext.setLogLevel("ERROR")

#     # Load all datasets from the training directory
#     print(f"\nLoading datasets from {training_data_dir}...")
#     datasets = {}
#     for sets in os.listdir(training_data_dir):
#         if sets.endswith('.parquet'):
#             dataset_name = sets.split('.')[0]  # Remove the .parquet extension
#             datasets[dataset_name] = spark.read.parquet(os.path.join(training_data_dir, sets))
#         else:
#             print(f"Skipping unsupported file format: {sets}")

#     for name, df in datasets.items():
#         print(f"Dataset: {name} with {df.count()} rows.")
    
#     # Prepare train, validation and test datasets
#     train_df = datasets['training']
#     validation_df = datasets['validation']
#     test_df = datasets['test']

#     # Assemble features into a single vector column
#     assembler = VectorAssembler(
#         inputCols=[col for col in train_df.columns if col not in ['Customer_ID', 'label', 'snapshot_date']], 
#         outputCol='features'
#     )
    
#     # Load Model Configuration
#     # base_model, param_grid = get_gbt_classifier() if model_type == 'gbt' else get_log_reg_classifier()


#     base_model, param_grid = get_gbt_classifier() if model_type == 'gbt' else get_log_reg_classifier()

#     # Prepare pipeline
#     pipeline = Pipeline(stages=[assembler, base_model])
#     evaluator = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderROC')

#     # Fit and tune the model
#     print("\nStarting model training and hyperparameter tuning...")
#     print(f"Size of parameter grid: {len(param_grid)} combinations")
#     results = []
#     for params in tqdm.tqdm(param_grid, desc="Training models", unit="model"):
#         # Set parameters for this run
#         for param, value in params.items():
#             base_model._set(**{param.name: value})

#         # Fit pipeline on train_df
#         model = pipeline.fit(train_df)
        
#         # Evaluate on train and validation data
#         train_predictions = model.transform(train_df)
#         train_auc = evaluator.evaluate(train_predictions)
#         val_predictions = model.transform(validation_df)
#         val_auc = evaluator.evaluate(val_predictions)
        
#         # Store results
#         results.append({
#             'params': {param.name: value for param, value in params.items()},
#             'metrics': {'train_auc': train_auc, 'val_auc': val_auc},
#             'model': model
#         })

#     # Find the best model by validation AUC & evaluating on test and OOT datasets
#     print("\nExtracting the best model based on validation AUC & evaluating on test & oot data...")
#     best_result = max(results, key=lambda x: x['metrics']['val_auc'])
#     best_model = best_result.pop('model')

#     test_predictions = best_model.transform(test_df)
#     best_result['metrics']['test_auc'] = evaluator.evaluate(test_predictions)

#     for i in range(1, len(datasets) - 2):
#         oot_df = datasets[f'oot_{i}']
#         oot_predictions = best_model.transform(oot_df)
#         best_result['metrics'][f'oot_{i}_auc'] = evaluator.evaluate(oot_predictions)

#     # Extract top 10 features based on feature importance
#     if model_type == 'gbt':
#         feature_importances = best_model.stages[-1].featureImportances.toArray()
#     else:
#         feature_importances = best_model.stages[-1].coefficients.toArray()
#     best_result['top_features'] = sorted(zip(assembler.getInputCols(), feature_importances), key=lambda x: x[1], reverse=True)[:10]

#     # Log results, features and model to mlflow
#     print("\nTraining completed. Best model:")
#     for set, metrics in best_result['metrics'].items():
#         print(f"{set}: {metrics}")
#     print("Top 10 features:")
#     for feature, importance in best_result['top_features']:
#         print(f"{feature}: {importance}")

#     # Save the best model
#     model_dir = os.path.join(model_store_dir, model_type)
#     model_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#     os.makedirs(model_dir, exist_ok=True)

#     # Save the corresponding metadata
#     best_model.save(os.path.join(model_dir, model_name))
#     with open(f"{os.path.join(model_dir, model_name)}_metadata.json", "w") as f:
#         json.dump(best_result, f, indent=4)
    
#     print(f"\nBest model saved to {os.path.join(model_dir, model_name)} and metadata saved to {os.path.join(model_dir, model_name)}_metadata.json")

#     # Stop Spark session
#     spark.stop()
#     print("\n\nModel training and logging completed successfully.\n\n")



import os
import argparse
import tqdm
from datetime import datetime
import json
import numpy as np

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# IMPORT: Import the necessary tuning tools from Spark ML
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Assuming these imports are correct based on your project structure
from configs.data import training_data_dir, model_store_dir
from configs.models import get_gbt_classifier, get_log_reg_classifier


if __name__ == "__main__":
    # Parse command line arguments to define model type
    parser = argparse.ArgumentParser(description='Train a machine learning model.')
    parser.add_argument('--model_type', type=str, choices=['gbt', 'logreg'], required=True, help='Type of model to train: gbt for Gradient Boosted Trees or logreg for Logistic Regression.')
    args = parser.parse_args()
    model_type = args.model_type

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Model Trainer") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.host", "127.0.0.1") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Load all datasets from the training directory
    print(f"\nLoading datasets from {training_data_dir}...")
    datasets = {}
    for sets in os.listdir(training_data_dir):
        if sets.endswith('.parquet'):
            dataset_name = sets.split('.')[0]  # Remove the .parquet extension
            datasets[dataset_name] = spark.read.parquet(os.path.join(training_data_dir, sets))
        else:
            print(f"Skipping unsupported file format: {sets}")

    for name, df in datasets.items():
        print(f"Dataset: {name} with {df.count()} rows.")
    
    # Prepare train, validation and test datasets
    train_df = datasets['training']
    # NOTE: CrossValidator uses its own internal validation, so the separate validation_df is used later for final reporting.
    validation_df = datasets['validation']
    test_df = datasets['test']

    # Assemble features into a single vector column
    assembler = VectorAssembler(
        inputCols=[col for col in train_df.columns if col not in ['Customer_ID', 'label', 'snapshot_date']], 
        outputCol='features'
    )
    
    # Load Model Configuration
    base_model, param_grid = get_gbt_classifier() if model_type == 'gbt' else get_log_reg_classifier()

    # Prepare pipeline
    pipeline = Pipeline(stages=[assembler, base_model])
    evaluator = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderROC')

    # --- REFACTORED: Use CrossValidator for efficient hyperparameter tuning ---
    print("\nSetting up CrossValidator for hyperparameter tuning...")
    num_folds = 3
    
    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=num_folds,  # Use 3-fold cross-validation. Reduce to 2 for more speed if needed.
        parallelism=4, # Set how many models to train in parallel
        seed=42
    )
    
    # --- ADDED: Enhanced Progress Logging ---
    total_models_to_train = len(param_grid) * num_folds
    print(f"\nStarting CrossValidator...")
    print(f"Will train across {len(param_grid)} parameter combinations and {num_folds} folds.")
    print(f"Total models to be trained: {total_models_to_train}")
    
    # This single .fit() call replaces the entire manual for loop.
    cv_model = crossval.fit(train_df)
    print("CrossValidator fitting complete.")

    # =========================================================================

    # Find the best model and its results
    print("\nExtracting the best model and evaluating on validation, test & oot data...")
    best_model = cv_model.bestModel
    
    # Get the parameters and metrics of the best model found by the CrossValidator
    best_params = best_model.stages[-1].extractParamMap()
    best_avg_val_auc = max(cv_model.avgMetrics) # The CrossValidator stores the average AUC for each param set

    # Create the results dictionary
    best_result = {
        'params': best_params,
        'metrics': {'avg_cv_validation_auc': best_avg_val_auc}
    }
    
    # Evaluate the single best model on all our hold-out sets
    for set_name, df in datasets.items():
        predictions = best_model.transform(df)
        auc = evaluator.evaluate(predictions)
        best_result['metrics'][f'{set_name}_auc'] = auc
        print(f"AUC for {set_name} dataset: {auc}")


    # Extract top 10 features based on feature importance
    try:
        final_model_stage = best_model.stages[-1]
        if hasattr(final_model_stage, 'featureImportances'):
             feature_importances = final_model_stage.featureImportances.toArray()
        elif hasattr(final_model_stage, 'coefficients'):
             feature_importances = final_model_stage.coefficients.toArray()
        else:
            feature_importances = None
        
        if feature_importances is not None:
            best_result['top_features'] = sorted(zip(assembler.getInputCols(), feature_importances), key=lambda x: x[1], reverse=True)[:10]
        else:
            best_result['top_features'] = "Not applicable for this model type"

    except Exception as e:
        print(f"Could not extract feature importances: {e}")
        best_result['top_features'] = "Error during extraction"


    # Log results, features and model
    print("\nTraining completed. Best model:")
    for set_name, metric_value in best_result['metrics'].items():
        print(f"{set_name}: {metric_value}")
    print("Top 10 features:")
    if isinstance(best_result['top_features'], list):
        for feature, importance in best_result['top_features']:
            print(f"{feature}: {importance}")
    else:
        print(best_result['top_features'])


    # Save the best model
    model_dir = os.path.join(model_store_dir, model_type)
    model_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(model_dir, exist_ok=True)

    # Save the corresponding metadata
    best_model.write().overwrite().save(os.path.join(model_dir, model_name))
    
    # Convert all complex objects in the results dictionary to simple,
    # JSON-serializable types (strings, floats, etc.) before saving.
    serializable_result = best_result.copy()
    serializable_result['params'] = {p.name: str(v) for p, v in best_result['params'].items()}
    # FIX: Correctly serialize the list of feature importance tuples
    if isinstance(best_result.get('top_features'), list):
        serializable_result['top_features'] = [{'feature': k, 'importance': float(v)} for k, v in best_result['top_features']]


    metadata_path = f"{os.path.join(model_dir, model_name)}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(serializable_result, f, indent=4)
    
    print(f"\nBest model saved to {os.path.join(model_dir, model_name)} and metadata saved to {metadata_path}")

    # Stop Spark session
    spark.stop()
    print("\n\nModel training and logging completed successfully.\n\n")
