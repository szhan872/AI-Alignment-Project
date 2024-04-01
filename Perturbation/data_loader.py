from datasets import load_dataset
import csv
import json
import re
import os
from random import Random
from typing import List

# Determine the current directory path (assumed to be where main.py resides)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the 'data' directory within the current directory
data_dir = os.path.join(current_dir, 'data')


def load_dataset_func(dataset_name, dataset_sub_name=None, original_columns=None, rename_columns=None):
    """
    Load specific columns from a dataset and optionally rename them.

    Parameters:
    - dataset_name: str, the name of the dataset.
    - dataset_sub_name: str, the sub name of the dataset (if applicable).
    - original_columns: list, list of original columns to keep.
    - rename_columns: dict, keys are original column names and values are the new column names.

    Returns:
    - A dataset with only the specified columns, with columns renamed as specified, or an error message.
    """
    try:
        # Load dataset; if a sub-name is provided, load that specific dataset
        if dataset_sub_name:
            dataset = load_dataset(dataset_name, dataset_sub_name)['train']
        else:
            dataset = load_dataset(dataset_name)['train']

        # If original columns are specified, remove all other columns
        if original_columns:
            dataset = dataset.remove_columns([col for col in dataset.column_names if col not in original_columns])

        # If rename_columns is specified, rename the columns accordingly
        if rename_columns:
            for original_col, new_col in rename_columns.items():
                if original_col != new_col:  # Only rename if the original and new names are different
                    dataset = dataset.rename_column(original_col, new_col)

        return dataset
    except Exception as e:
        return f"Error loading dataset: {str(e)}"


def load_white_american_first_names(data_dir=data_dir, file_name="/person_name.txt"):
    """
    Load white American first names from a specified file.

    Args:
        data_dir (str): Directory path where the names file is located.
        file_name (str): The name of the file containing the names.

    Returns:
        A list of white American first names.
    """

    file_path = f"{data_dir}{file_name}"
    white_american_names = []
    with open(file_path, encoding="utf-8") as f:
        for line in f.readlines():
            parts = line.replace("\n", "").split(",")
            if len(parts) < 4:
                continue  # Skip lines that do not have the expected format
            name, name_type, category, value = parts[0], parts[1], parts[2], parts[3]
            if name_type.lower() == "first_name" and category.lower() == "race" and value.lower() == "white_american":
                white_american_names.append(name.strip().lower())
    return white_american_names



def load_dialect_mapping(data_dir=data_dir, file_name="/SAE_to_AAVE_mapping.json"):
    """
    Load the dialect mapping from a JSON file.

    Args:
        data_dir (str): Directory path where the mapping file is located.
        file_name (str): The name of the file containing the dialect mappings.

    Returns:
        A list of keys from the dialect mapping.
    """

    dialect_mapping_file_path = f"{data_dir}{file_name}"
    with open(dialect_mapping_file_path, 'r') as file:
        dialect_mapping = json.load(file)
    return list(dialect_mapping.keys())


def contains_and_highlight_terms(text, terms):
    """
    Check if the text contains any of the specified terms and highlight them.

    Args:
        text (str): The text to search within.
        terms (List[str]): The list of terms to search for.

    Returns:
        A tuple (found, highlighted_text), where 'found' is a boolean indicating if any term was found,
        and 'highlighted_text' is the text with the terms highlighted.
    """

    highlighted_text = text
    found = False
    for term in terms:
        # Match complete words only using regular expressions
        pattern = r"\b{}\b".format(re.escape(term))
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            found = True
            # Highlight each matched term
            for match in set(matches):  # Use set to avoid duplicating highlights
                highlighted_text = re.sub(pattern, f"【{match}】", highlighted_text, flags=re.IGNORECASE)
                print(highlighted_text)
    return found, highlighted_text


