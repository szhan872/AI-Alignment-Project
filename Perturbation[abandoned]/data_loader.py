from datasets import load_dataset
import csv
import json
import re
import os
from random import Random
from typing import List

# 获取当前文件（main.py）的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建 data 文件夹的路径
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
        if dataset_sub_name:
            dataset = load_dataset(dataset_name, dataset_sub_name)['train']
        else:
            dataset = load_dataset(dataset_name)['train']

        if original_columns:
            # Removing unwanted columns
            dataset = dataset.remove_columns([col for col in dataset.column_names if col not in original_columns])

        if rename_columns:
            # Renaming specified columns
            for original_col, new_col in rename_columns.items():
                if original_col != new_col:  # Only rename if the original and new names are different
                    dataset = dataset.rename_column(original_col, new_col)

        return dataset
    except Exception as e:
        return f"Error loading dataset: {str(e)}"

# Example usage
# dataset_name = 'ought/raft'
# dataset_sub_name = 'tweet_eval_hate'
# sentence_name = 'Tweet'
# label_name = 'Label'
# original_columns = [sentence_name, label_name]
# rename_columns = {sentence_name: 'Sentence', label_name: 'Label'}

# Load dataset and rename columns
# dataset_text = load_dataset_func(dataset_name, dataset_sub_name, original_columns, rename_columns)

# name_file_path = f"{root_path}person_name.txt"
# dialect_mapping_file_path = f'{root_path}SAE_to_AAVE_mapping.json'


def load_white_american_first_names(data_dir=data_dir, file_name="/person_name.txt"):
    file_path = f"{data_dir}{file_name}"
    white_american_names = []
    with open(file_path, encoding="utf-8") as f:
        for line in f.readlines():
            parts = line.replace("\n", "").split(",")
            if len(parts) < 4:
                continue  # Skip lines that don't have enough parts
            name, name_type, category, value = parts[0], parts[1], parts[2], parts[3]
            if name_type.lower() == "first_name" and category.lower() == "race" and value.lower() == "white_american":
                white_american_names.append(name.strip().lower())
    return white_american_names



def load_dialect_mapping(data_dir=data_dir, file_name="/SAE_to_AAVE_mapping.json"):
    # Load the JSON file
    dialect_mapping_file_path = f"{data_dir}{file_name}"
    with open(dialect_mapping_file_path, 'r') as file:
        dialect_mapping = json.load(file)
    return list(dialect_mapping.keys())


def contains_and_highlight_terms(text, terms):
    highlighted_text = text
    found = False
    for term in terms:
        # 使用正则表达式匹配完整的单词
        pattern = r"\b{}\b".format(re.escape(term))
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            found = True
            for match in set(matches):  # 使用set去重，避免重复替换相同的词
                highlighted_text = re.sub(pattern, f"【{match}】", highlighted_text, flags=re.IGNORECASE)
                print(highlighted_text)
    return found, highlighted_text


