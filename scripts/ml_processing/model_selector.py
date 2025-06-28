import os
import json

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

from configs.data import model_store_dir


def load_metrics(metrics_path):
    with open(metrics_path, "r") as f:
        return json.load(f)


def select_best_model(models_dir, criterion="val_auc"):
    best_score = 0
    best_model = None
    best_metadata = None

    for model_name in os.listdir(models_dir):
        model_sub_dir = os.path.join(models_dir, model_name)
        
        for element in os.listdir(model_sub_dir):
            if os.path.isdir(os.path.join(model_sub_dir, element)):
                # Get the model and metadata paths
                model_path = os.path.join(model_sub_dir, element)
                metadata_path = os.path.join(model_sub_dir, f'{element}_metadata.json')

                # Load model and metadata
                print(f"Loading model: {model_name} from {model_path} and metadata from {metadata_path}")
                model = PipelineModel.load(model_path)
                metadata = load_metrics(metadata_path)

                score = metadata['metrics'].get(criterion, 0)
                print(f"Model: {model_name}, {criterion}: {score}")

                # Check performance metrics
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_metadata = metadata

            elif element.endswith('metadata.json'):
                continue
            else:
                raise ValueError(f"Unexpected file structure in {model_sub_dir}")

    if best_model and best_metadata:
        print(f"Best model: {best_model}")
        print("Metrics:", best_metadata["metrics"])
        return best_model, best_metadata
    else:
        raise ValueError("No valid models found in the registry directory.")


if __name__ == "__main__":
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Model Selection") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Load all models from the model registry directory
    print(f"\nLoading models from {model_store_dir}...")
    if not os.path.exists(model_store_dir):
        raise FileNotFoundError(f"Model registry directory {model_store_dir} does not exist.")
    
    # Iterate through all model directories and select the best model based on validation AUC
    best_model, best_info = select_best_model(model_store_dir)

    # Save the best model if found in best registry
    if best_model and best_info:
        best_model.save(os.path.join(model_store_dir, "best_model"))
        with open(os.path.join(model_store_dir, "best_model", "metrics.json"), "w") as f:
            json.dump(best_info["metrics"], f, indent=4)
    
        print(f"Best model saved to {os.path.join(model_store_dir, 'best_model')}")
    else:
        print("No best model found to save.")
    
    # Stop the Spark session
    spark.stop()
    print("Spark session stopped.")