import os
import csv
import re
from datasets import load_dataset
from random import Random
from typing import Optional
import os
from data_loader import load_white_american_first_names, load_dialect_mapping, contains_and_highlight_terms
from datasets.utils.py_utils import NonMutableDict

# Load specific sets of names and dialect terms for perturbation
white_american_first_names = load_white_american_first_names()
dialect_map_keys = load_dialect_mapping()

# Mapping terms for each type of perturbation
terms_maps = {
    'gender': [
        # Gender-specific terms for replacement or highlighting
        "son", "sons", "boy", "boys", "man", "men", "father", "fathers", "brother", "brothers",
        "nephew", "nephews", "king", "kings", "waiter", "waiters", "dad", "dads", "stepson",
        "stepsons", "stepfather", "stepfathers", "grandson", "grandsons", "grandfather",
        "grandfathers", "male", "males", "he", "him", "his", "himself"
    ],

    'race': white_american_first_names,     # Race-specific names for perturbation
    'dialect': dialect_map_keys             # Dialect-specific terms for perturbation
}

def extract_and_write_terms(dataset, safe_dataset_name, original_type, perturb_type, max_count=100,
                            task_type='text classification', filter_sentence_flag=True, balance_label_flag=True,
                            label_cat_1=None, label_cat_2=NonMutableDict,
                            root_path: Optional[str] = None,
                            dataset_path: Optional[str] = None):
    """
    Extracts sentences from the dataset and writes them to a CSV file after applying
    specified perturbations and filters.

    Parameters:
    - dataset: The dataset to process.
    - safe_dataset_name: The dataset name, formatted to be filesystem safe.
    - original_type: The type of perturbation to apply.
    - perturb_type: The specific perturbation detail.
    - max_count: Maximum number of sentences to include.
    - task_type: The type of NLP task (e.g., text classification).
    - filter_sentence_flag: If true, only include sentences containing specific terms.
    - balance_label_flag: If true, balance the number of instances for each label.
    - label_cat_1, label_cat_2: The categories for classification.
    - root_path: The root directory for saving output.
    - dataset_path: The specific path under the root directory for the dataset.
    """

    print('task_type:', task_type)
    print('original_type:', original_type)
    print('perturb_type:', perturb_type)
    print('filter_sentence_flag:', filter_sentence_flag)
    print('balance_label_flag:', balance_label_flag)

    # Set default paths if not provided
    if root_path is None:
        root_path = os.path.dirname(os.path.abspath(__file__))
    if dataset_path is None:
        dataset_path = os.path.join(root_path, 'datasets')

    # Ensure the directory exists
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    directory_path = dataset_path
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Adjust max_count based on the actual dataset size
    dataset_size = len(dataset)
    if dataset_size < max_count:
        print(
            f"Dataset size ({dataset_size}) is smaller than max_count ({max_count}). Adjusting max_count accordingly.")
        max_count = dataset_size

    # Adjust the number of instances per label if balancing is enabled
    max_per_label = max_count // 2 if balance_label_flag else max_count
    print(f"max_per_label: {max_per_label}")

    collected_sentences = []
    true_count = 0
    false_count = 0
    terms = terms_maps[original_type]

    for example in dataset:
        if len(collected_sentences) >= max_count:
            break
        if balance_label_flag and true_count >= max_per_label and false_count >= max_per_label:
            break   # Stop if the count for each label has reached max_per_label

        # Process label based on task type
        if task_type == 'toxicity detection':
            label = 1 if example['Label'] > 0.5 else 0
        else:
            label_cat_1 = str(label_cat_1)
            label_cat_2 = str(label_cat_2)
            label = str(example['Label'])

            if label == label_cat_1:
                label = 0
            elif label == label_cat_2:
                label = 1
            else:
                print(f"Unknown label: {label}")
                continue  # Skip if the label is not recognized

        highlighted_text = example['Sentence']

        # Highlight terms in sentences if filtering is enabled
        if filter_sentence_flag:
            found, _ = contains_and_highlight_terms(example['Sentence'], terms)
            if not found:
                continue  # Skip this sentence if no terms are found

        # Add the sentence to the collection, respecting balance flags
        if not balance_label_flag or (label == 1 and true_count < max_per_label) or (
                label == 0 and false_count < max_per_label):
            collected_sentences.append((highlighted_text, label))
            if label == 1:
                true_count += 1
            else:
                false_count += 1
        print(f"true_count: {true_count}, false_count: {false_count}")

    # Write the collected sentences to a CSV file
    csv_file_path = f'{directory_path}/original_{original_type}_{safe_dataset_name}_{perturb_type}.csv'
    print(csv_file_path)
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Sentence', 'Label'])
        for sentence, label in collected_sentences:
            writer.writerow([sentence, label])

    print(f"Total lines included: {len(collected_sentences)}")

