
from src import utils
import os
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler


    # for i, col in enumerate(df.columns[:5]):
    #     fig, ax = plt.subplots(1)
    #     ax.hist(df[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black', label=f'{col} Distribution')
    #     # Add title, labels, legend, and grid
    #     ax.set_title(f"Histogram of {col}", fontsize=14, fontweight='bold')
    #     ax.set_xlabel(f"{col}", fontsize=12)
    #     ax.set_ylabel("Frequency", fontsize=12)
    #     ax.legend(loc="upper right", fontsize=10)
    #     ax.grid(True, linestyle="--", alpha=0.6)
        
    #     plt.tight_layout()  # Adjust layout for better display
    # plt.show()






def already_processed(output_file, experiment_config_name, model_config_name):
    """Check if the experiment and model configuration has already been processed."""
    if not os.path.exists(output_file):
        return False

    try:
        df = pd.read_csv(output_file)
    except pd.errors.EmptyDataError:
        # File exists but is empty
        return False

    return not df[(df['experiment_config_name'] == experiment_config_name) &
                  (df['model_config_name'] == model_config_name)].empty



def main():
    print('Hello world')

    # STEP 1: Split dataset into cross-validation and heldout splits.
    # data_handler.generate_crossval_heldout_splits()
    config = utils.get_config()

    output_file = config['results_path']
    fieldnames = [
        "experiment_config_name",
        "model_config_name",
        "f1-score_mean", "f1-score_std",
        "precision_mean", "precision_std",
        "recall_mean", "recall_std",
        "accuracy_mean", "accuracy_std"
    ]

    file_mode = "a" if os.path.exists(output_file) else "w"

    with open(output_file, mode=file_mode, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header only if the file is newly created
        if file_mode == "w":
            writer.writeheader()

        for experiment_config_name in utils.get_all_experiment_configuration_names():
            experiment_configuration = utils._get_experiment_configuration_from_name(experiment_config_name)
            X, y = utils.matrices_from_configuration_name(experiment_config_name)

            for model_config_name in utils.get_all_model_config_names():
                utils.io.info(f'Running {experiment_config_name}, {model_config_name}')
                model = utils.model_from_config_name(model_config_name)

                if already_processed(output_file, experiment_config_name, model_config_name):
                    print(f"Skipping {experiment_config_name} with {model_config_name}: Already processed.")
                    continue


                kf = KFold(n_splits=3)

                metrics = {
                    'f1-score': [],
                    'precision': [],
                    'recall': [],
                    'accuracy': [],
                }

                for train_index, test_index in kf.split(X):
                    # Split the data
                    X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values
                    y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values

                    if 'undersampling' in experiment_configuration and experiment_configuration['undersampling']:
                        rus = RandomUnderSampler(random_state=42)
                        X_train, y_train = rus.fit_resample(X_train, y_train)

                    model.fit(X_train, y_train)

                    # Test the model
                    y_pred = model.predict(X_test)

                    # Evaluate accuracy
                    metrics['f1-score'].append(f1_score(y_test, y_pred))
                    metrics['precision'].append(precision_score(y_test, y_pred))
                    metrics['recall'].append(recall_score(y_test, y_pred))
                    metrics['accuracy'].append(accuracy_score(y_test, y_pred))

                for metric, value in metrics.items():
                    utils.io.info(f'{metric:9}: {np.average(value):4.3f}')
                # Compute mean and standard deviation for each metric
                results = {
                    "experiment_config_name": experiment_config_name,
                    "model_config_name": model_config_name,
                }

                for metric, values in metrics.items():
                    results[f"{metric}_mean"] = np.mean(values)
                    results[f"{metric}_std"] = np.std(values)

                # Write the results to the CSV file
                writer.writerow(results)
  
    
if __name__ == '__main__':
    main()
    # for model_config_name in utils.get_all_model_config_names():
    #     print(utils.model_from_config_name(model_config_name))

    # for experiment_config_name in utils.get_all_experiment_configuration_names():
    #     utils.matrices_from_configuration_name(experiment_config_name)