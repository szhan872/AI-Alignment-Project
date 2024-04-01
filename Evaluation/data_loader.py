# Import necessary libraries for data handling

import pandas as pd
import os


def load_dataset(safe_dataset_name, dataset_path, perturb_type, perturb_detail):
    """
    Loads the original and perturbed datasets from CSV files.

    Parameters:
    - safe_dataset_name: A version of the dataset name that is safe for use in file paths (e.g., no special characters).
    - dataset_path: The base directory where the datasets are stored.
    - perturb_type: The type of perturbation applied (e.g., 'gender', 'race'). This is used to construct the file names.
    - perturb_detail: Specific detail about the perturbation (e.g., 'male_to_female') used in file naming.

    Returns:
    - original_dataset: DataFrame containing the original data.
    - perturbed_dataset: DataFrame containing the perturbed data.
    - original_dataset_path: The file path to the original dataset.
    - perturbed_dataset_path: The file path to the perturbed dataset.
    """

    # Construct file paths for the original and perturbed datasets
    original_dataset_path = os.path.join(dataset_path,
                                         f'original_{perturb_type}_{safe_dataset_name}_{perturb_detail}.csv')
    perturbed_dataset_path = os.path.join(dataset_path,
                                            f'perturbed_{perturb_type}_{safe_dataset_name}_{perturb_detail}.csv')

    # Load datasets from the constructed paths
    original_dataset = pd.read_csv(original_dataset_path)
    perturbed_dataset = pd.read_csv(perturbed_dataset_path)

    # Print paths to the console for verification
    print("Original dataset path: ", original_dataset_path)
    print("Perturb dataset path: ", perturbed_dataset_path)

    # Print the first few rows of each dataset for a quick overview
    print("First few rows of the original dataset:\n", original_dataset.head())
    print("First few rows of the perturbed dataset:\n", perturbed_dataset.head())

    # Return the loaded datasets along with their paths
    return original_dataset, perturbed_dataset, original_dataset_path, perturbed_dataset_path

