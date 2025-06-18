#############
## Imports ##
#############

import pickle

import bnlearn as bn
import numpy as np
import pandas as pd
from test_model import test_model

######################
## Boilerplate Code ##
######################


def load_data():
    """Load train and validation datasets from CSV files."""
    # Implement code to load CSV files into DataFrames
    # Example: train_data = pd.read_csv("train_data.csv")
    df_data = pd.read_csv("train_data.csv")
    df_val = pd.read_csv("validation_data.csv")

    return df_data, df_val


def make_network(df):
    """Define and fit the initial Bayesian Network."""
    # Code to define the DAG, create and fit Bayesian Network, and return the model

    edges = [
        ("Start_Stop_ID", "End_Stop_ID"),
        ("Start_Stop_ID", "Zones_Crossed"),
        ("Start_Stop_ID", "Route_Type"),
        ("Start_Stop_ID", "Fare_Category"),
        ("Start_Stop_ID", "Distance"),
        ("End_Stop_ID", "Zones_Crossed"),
        ("End_Stop_ID", "Route_Type"),
        ("End_Stop_ID", "Fare_Category"),
        ("End_Stop_ID", "Distance"),
        ("Distance", "Zones_Crossed"),
        ("Distance", "Route_Type"),
        ("Distance", "Fare_Category"),
        ("Zones_Crossed", "Route_Type"),
        ("Zones_Crossed", "Fare_Category"),
        ("Route_Type", "Fare_Category"),
    ]

    bn_DAG = bn.make_DAG(edges)
    bn.plot(bn_DAG)

    bn_model = bn.parameter_learning.fit(bn_DAG, df)

    return bn_model


def make_pruned_network(df):
    """Define and fit a pruned Bayesian Network."""
    # Code to create a pruned network, fit it, and return the pruned model

    edges = [
        ("Start_Stop_ID", "Zones_Crossed"),
        ("Start_Stop_ID", "Distance"),
        ("End_Stop_ID", "Zones_Crossed"),
        ("End_Stop_ID", "Distance"),
        ("Distance", "Fare_Category"),
        ("Zones_Crossed", "Fare_Category"),
        ("Route_Type", "Fare_Category"),
    ]

    pruned_DAG = bn.make_DAG(edges)
    bn.plot(pruned_DAG)

    pruned_model = bn.parameter_learning.fit(pruned_DAG, df)

    return pruned_model


def make_optimized_network(df):
    """Perform structure optimization and fit the optimized Bayesian Network."""
    # Code to optimize the structure, fit it, and return the optimized model

    final_DAG = bn.structure_learning.fit(df, methodtype="hc", scoretype="bic")
    bn.plot(final_DAG)

    final_model = bn.parameter_learning.fit(final_DAG, df)

    return final_model


def save_model(fname, model):
    """Save the model to a file using pickle."""
    f = open(fname, "wb")
    pickle.dump(model, f)
    f.close()


def load_model(fname):
    with open(fname, "rb") as f:
        model = pickle.load(f)
    return model


def evaluate(model_name, val_df):
    """Load and evaluate the specified model."""
    with open(f"{model_name}.pkl", "rb") as f:
        model = pickle.load(f)
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        print(f"Total Test Cases: {total_cases}")
        print(f"Total Correct Predictions: {correct_predictions} out of {total_cases}")
        print(f"Model accuracy on filtered test cases: {accuracy:.2f}%")


############
## Driver ##
###########


def main():
    # Load data
    train_df, val_df = load_data()

    # Create and save base model
    # base_model = make_network(train_df)
    # save_model("base_model.pkl", base_model)

    # Create and save pruned model
    # pruned_network = make_pruned_network(train_df)
    # save_model("pruned_model.pkl", pruned_network)

    # Create and save optimized model
    # optimized_network = make_optimized_network(train_df)
    # save_model("optimized_model.pkl", optimized_network)

    # Evaluate all models on the validation set
    evaluate("base_model", val_df)
    evaluate("pruned_model", val_df)
    evaluate("optimized_model", val_df)

    print("[+] Done")


if __name__ == "__main__":
    main()
