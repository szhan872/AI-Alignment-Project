import csv
import re
import os
from random import Random
from pathlib import Path
from name_perturbation import PersonNamePerturbation
from gender_perturbation import GenderPerturbation
from dialect_perturbation import DialectPerturbation

# 获取当前文件（main.py）的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建 data 文件夹的路径
data_dir = os.path.join(current_dir, 'data')

def apply_perturbation_and_save(root_path, dataset_path, choice_number, original_type, perturb_type, safe_dataset_name):
    """
    Apply perturbation based on the choice number and save the perturbated sentences to a new CSV file.

    Args:
        dataset (list): List of sentences and labels to be perturbated.
        root_path (str): The root directory where files are located and to be saved.
        choice_number (int): Index to choose the perturbation type.
        original_type (str): Original type of perturbation (e.g., 'gender').
        perturb_type (str): Specific perturbation to apply (e.g., 'male_to_female').
        safe_dataset_name (str): A safe version of the dataset name for file naming.
    """

    rng = Random()

    ### NAME PERTURBATION

    directory_path = dataset_path
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    person_name_file_path = f"{data_dir}/person_name.txt"

    name_perturbation = PersonNamePerturbation(
        prob=1.0,
        source_class={"race": "white_american"},
        target_class={"race": "black_american"},
        person_name_type="first_name",
        preserve_gender=True,
        name_file_path=person_name_file_path
    )

    ## GENDER PERTURBATION

    # # Initialize the GenderPerturbation instance
    gender_perturbation = GenderPerturbation(mode="terms", prob=1.0, source_class="male", target_class="female")

    name_substitution_mapping = {}
    skipped_tokens = set()

    ## DIALECT PERTURBATION

    dialect_mapping_file_path = f'{data_dir}/SAE_to_AAVE_mapping.json'
    dialect_perturbations = DialectPerturbation(prob=1.0, source_class="SAE", target_class="AAVE",
                                                mapping_file_path=dialect_mapping_file_path)

    # 创建扰动方法的映射，注意对于 name_perturbation 特殊处理
    perturb_methods = {
        'name_perturbation': lambda sentence, rng: name_perturbation.perturb_with_persistency(sentence, rng,
                                                                                              name_substitution_mapping,
                                                                                              skipped_tokens),
        'gender_perturbation': lambda sentence, rng: gender_perturbation.perturb(sentence, rng),
        'dialect_perturbation': lambda sentence, rng: dialect_perturbations.perturb(sentence, rng)
    }

    # 选择扰动方法
    perturb_method = perturb_methods[list(perturb_methods.keys())[choice_number]]

    original_csv_path = f'{directory_path}/original_{original_type}_{safe_dataset_name}_{perturb_type}.csv'
    perturbated_sentences = []
    perturbated_labels = []

    print('dataset_name: ', safe_dataset_name)
    print('original_type', original_type)
    print('perturb_type', perturb_type)

    # Function to normalize sentences for comparison
    def normalize_sentence(sentence):
        return re.sub(r'[^\w\s]', '', sentence).lower().strip()

    # Apply perturbation
    with open(original_csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header

        for row in reader:
            sentence, label = row
            perturbated_sentence = perturb_method(sentence, rng)
            perturbated_sentences.append(perturbated_sentence)
            perturbated_labels.append(label)

            # Normalize sentences for comparison
            normalized_original = normalize_sentence(sentence)
            normalized_perturbated = normalize_sentence(perturbated_sentence)

            # Check if the normalized sentences differ
            if normalized_original != normalized_perturbated:
                print("Original Sentence   :", sentence)
                print("Perturbated Sentence:", perturbated_sentence)
                print("Label               :", label)
                print()  # Print a blank line for better readability

    # Save perturbated sentences to a new CSV file
    perturb_file_path = f'{directory_path}/perturbed_{original_type}_{safe_dataset_name}_{perturb_type}.csv'
    with open(perturb_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Sentence', 'Label'])
        for sentence, label in zip(perturbated_sentences, perturbated_labels):
            writer.writerow([sentence, label])

    print(f"Perturbation applied and saved to {perturb_file_path}")

