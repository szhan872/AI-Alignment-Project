import csv
import re
import os
from random import Random
from pathlib import Path
from name_perturbation import PersonNamePerturbation
from gender_perturbation import GenderPerturbation
from dialect_perturbation import DialectPerturbation

# Determine the current directory path (assumed to be where main.py resides)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the 'data' directory within the current directory
data_dir = os.path.join(current_dir, 'data')

def apply_perturbation_and_save(root_path, dataset_path, choice_number, original_type, perturb_type, safe_dataset_name):
    """
    Applies a specified perturbation to sentences from an original dataset and saves the perturbed sentences to a new CSV file.

    Parameters:
        root_path (str): The root directory where output files are saved.
        dataset_path (str): The specific path within the root directory where the dataset resides.
        choice_number (int): The index indicating the chosen perturbation method.
        original_type (str): The type of the original perturbation (e.g., 'gender').
        perturb_type (str): The specific perturbation to apply (e.g., 'male_to_female').
        safe_dataset_name (str): A filesystem-safe version of the dataset name for file naming.
    """

    rng = Random()

    # Ensure the directory exists where files will be saved
    directory_path = dataset_path
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


    ### NAME PERTURBATION SETUP
    # Path to a file containing person names for perturbation
    person_name_file_path = f"{data_dir}/person_name.txt"

    # Initialize the PersonNamePerturbation instance with predefined settings
    name_perturbation = PersonNamePerturbation(
        prob=1.0,
        source_class={"race": "white_american"},
        target_class={"race": "black_american"},
        person_name_type="first_name",
        preserve_gender=True,
        name_file_path=person_name_file_path
    )

    ### GENDER PERTURBATION SETUP
    # Initialize the GenderPerturbation instance with predefined settings
    gender_perturbation = GenderPerturbation(mode="terms", prob=1.0, source_class="male", target_class="female")

    # Stores name substitutions to ensure consistency across perturbations
    name_substitution_mapping = {}
    # Tokens that are skipped during perturbation
    skipped_tokens = set()

    ### DIALECT PERTURBATION SETUP
    # Path to a file containing dialect mappings for perturbation
    dialect_mapping_file_path = f'{data_dir}/SAE_to_AAVE_mapping.json'
    # Initialize the DialectPerturbation instance with predefined settings
    dialect_perturbations = DialectPerturbation(prob=1.0, source_class="SAE", target_class="AAVE",
                                                mapping_file_path=dialect_mapping_file_path)

    # Map perturbation methods to callable functions for easy access
    perturb_methods = {
        'name_perturbation': lambda sentence, rng: name_perturbation.perturb_with_persistency(sentence, rng,
                                                                                              name_substitution_mapping,
                                                                                              skipped_tokens),
        'gender_perturbation': lambda sentence, rng: gender_perturbation.perturb(sentence, rng),
        'dialect_perturbation': lambda sentence, rng: dialect_perturbations.perturb(sentence, rng)
    }

    # Select the perturbation method based on the choice number
    perturb_method = perturb_methods[list(perturb_methods.keys())[choice_number]]

    # Path to the original dataset CSV file
    original_csv_path = f'{directory_path}/original_{original_type}_{safe_dataset_name}_{perturb_type}.csv'
    # Prepare to collect perturbed sentences and labels
    perturbated_sentences = []
    perturbated_labels = []

    print('dataset_name: ', safe_dataset_name)
    print('original_type', original_type)
    print('perturb_type', perturb_type)

    # Helper function to normalize sentences for comparison
    def normalize_sentence(sentence):
        return re.sub(r'[^\w\s]', '', sentence).lower().strip()

    # Apply perturbation and collect results
    with open(original_csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            sentence, label = row
            perturbated_sentence = perturb_method(sentence, rng)
            perturbated_sentences.append(perturbated_sentence)
            perturbated_labels.append(label)

            # Normalize and compare original and perturbated sentences
            normalized_original = normalize_sentence(sentence)
            normalized_perturbated = normalize_sentence(perturbated_sentence)
            if normalized_original != normalized_perturbated:
                print("Original Sentence   :", sentence)
                print("Perturbated Sentence:", perturbated_sentence)
                print("Label               :", label)
                print()  # Print a blank line for better readability

    # Save perturbated sentences and labels to a new CSV file
    perturb_file_path = f'{directory_path}/perturbed_{original_type}_{safe_dataset_name}_{perturb_type}.csv'
    with open(perturb_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Sentence', 'Label'])
        for sentence, label in zip(perturbated_sentences, perturbated_labels):
            writer.writerow([sentence, label])

    print(f"Perturbation applied and saved to {perturb_file_path}")

