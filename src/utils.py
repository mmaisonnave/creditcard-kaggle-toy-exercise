import os
import yaml
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from src import data_handler
from sklearn.preprocessing import StandardScaler
from src import data_handler
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
def get_config() -> dict:
    """
    Load and parse the configuration file.

    This function reads a YAML configuration file and 
    returns its contents as a Python dictionary. If the file is not found, 
    a FileNotFoundError is raised with a descriptive error message.

    Returns:
        dict: A dictionary containing the configuration data.

    Raises:
        FileNotFoundError: If the configuration file does not exist at the specified path.
        yaml.YAMLError: If there is an error while parsing the YAML file.
    """
    config_path = "config/paths.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    return config

class io:
    def info(message: str) -> None:
        print(f'[ INFO  ] {message}')

    def error(message: str) -> None:
        print(f'[ ERROR ] {message}')

    def warning(message: str) -> None:
        print(f'[WARNING] {message}')

    def ok(message: str) -> None:
        print(f'[  OK   ] {message}')

def exploratory_data_analysis(df):
    io.info(f'Number of instances={df.shape[0]:,}')
    io.info(f'Number of variables={df.shape[1]}')


    for i, col in enumerate(df.columns):
        io.info(f'#{i+1:>2} Column name={col:6} '\
        f'(dtype={str(df[col].dtype):7}) ' \
            f'(NA count={df[col].isna().sum()})' \
            f'(Average={np.average(df[col]):9.3f}) ' \
            f'(STD={np.std(df[col]):9.3f}) '
            )

    io.info(f"Number of frauds {np.sum(df['Class'])}/{len(df['Class']):,} ({100 * (np.sum(df['Class']) / len(df['Class'])):.2f}%)")


def get_all_model_config_names():
    config = get_config()
    yaml_path = os.path.join(config['repository_path'], config['model_config_path'])   
    with open(yaml_path, "r") as file:
        model_configs = yaml.safe_load(file)
    return model_configs.keys()

def model_from_config_name(config_name):
    config = get_config()
    yaml_path = os.path.join(config['repository_path'], config['model_config_path'])   
    with open(yaml_path, "r") as file:
        model_configs = yaml.safe_load(file)
    return _model_from_config(model_configs[config_name])


def _model_from_config(model_config):
    if model_config['model_type']=='LogisticRegression':
        return LogisticRegression(**model_config['params'])
    elif model_config['model_type']=='RandomForestClassifier':
        return RandomForestClassifier(**model_config['params'])
    elif model_config['model_type']=='SVC':
        return SVC(**model_config['params'])
    elif model_config['model_type']=='BalancedRandomForestClassifier':
        return BalancedRandomForestClassifier(**model_config['params'])
    elif model_config['model_type']=='GradientBoostingClassifier':
        return GradientBoostingClassifier(**model_config['params'])
    elif model_config['model_type']=='KNeighborsClassifier':
        return KNeighborsClassifier(**model_config['params'])
    else:
        raise ValueError(f"Unsupported model name: {model_config['model_type']}")

def get_all_experiment_configuration_names():
    config = get_config()
    yaml_path = os.path.join(config['repository_path'], config['experiment_config_path'])   
    with open(yaml_path, "r") as file:
        experiment_configs = yaml.safe_load(file)
    return experiment_configs.keys()

def _get_experiment_configuration_from_name(experiment_configuration_name):
    config = get_config()
    yaml_path = os.path.join(config['repository_path'], config['experiment_config_path'])   
    with open(yaml_path, "r") as file:
        experiment_configs = yaml.safe_load(file)
    return experiment_configs[experiment_configuration_name]


def matrices_from_configuration_name(experiment_configuration_name):
    params = _get_experiment_configuration_from_name(experiment_configuration_name)
    # Step #1: Load Data
    # Step #2: Process
    # Step #3: return X, y
    # STEP 2: Retrieve cross-validation split
    crossval_df = data_handler.get_data(split='crossval')

    # STEP 3: Preprocessing:
    if params['normalization']:
        scaler = StandardScaler()
        crossval_df['Amount'] = scaler.fit_transform(crossval_df[['Amount']])
        for ix in range(1,29):
            crossval_df[f'V{ix}'] = scaler.fit_transform(crossval_df[[f'V{ix}']])
    
    # STEP 4: Exploratory analysis
    # exploratory_data_analysis(crossval_df)

    # STEP 5: Cross-validation

    # For each model_config in model_configs.yaml:
    #   run K-fold CV
    #   model = model_from_config(model_config)
    #   Store results in results.csv (including name of model config)



    X = crossval_df[[f'V{i}' for i in range(1,29)]+['Amount']]
    y = crossval_df['Class']

    return X, y


