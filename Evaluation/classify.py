import torch
import json
import re
import pandas as pd
import csv
import time
import os

def classify_sentence(sentence, cat_1, cat_2, model, tokenizer):
    """
    Classify a sentence into one of two categories using a given model and tokenizer,
    and return the model's output as a formatted string.

    Parameters:
    - sentence: The text sentence to classify.
    - cat_1, cat_2: The two categories for classification.
    - model: The pre-loaded model used for classification.
    - tokenizer: The tokenizer corresponding to the model.

    Returns:
    A string containing the model's classification output.
    """

    # Automatically select the device (GPU or CPU) based on availability
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically selects CUDA if available, else CPU

    # Construct the prompt with instructions for classification
    prompt = f'<s> [INST] Given the sentence "{sentence}", classify it as either "{cat_1}" or "{cat_2}". Provide the probability that the sentence belongs to "{cat_1}", regardless of the predicted category. Format the response in JSON with "category" for the predicted classification and "probability" for the likelihood of "{cat_1}" [/INST]</s>'

    # Tokenize the prompt and move it to the selected device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate output from the model
    outputs = model.generate(**inputs, max_new_tokens=512)

    # Decode the output to a string and return it
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_json(output_str):
    """
    Extracts JSON formatted text from a string using regular expression.

    Parameters:
    - output_str: The string output from which to extract JSON.

    Returns:
    - category: The classified category extracted from JSON.
    - probability: The probability associated with the category.
    """

    # Try to find a JSON string within the output
    json_match = re.search(r'\{.*?\}', output_str.replace("\n", ""), re.DOTALL)
    if json_match:
        json_str = json_match.group(0)  # Extract the JSON string
        try:
            # Parse the JSON string
            json_data = json.loads(json_str)
            # Extract category and probability from the JSON data
            category = json_data.get("category", None)
            probability = json_data.get("probability", 0)
            return category, probability
        except json.JSONDecodeError:
            print("Error decoding JSON from the matched string.")
            return None, 0
    else:
        print("No JSON format found in the result.")
        return None, 0


def map_category_to_number(category, cat_1, cat_2):
    """
    Maps a category label to a numerical value.

    Parameters:
    - category: The category to map.
    - cat_1, cat_2: The two categories for comparison.

    Returns:
    - A numerical representation of the category (0, 1, or -1 for unknown).
    """

    # Map categories to numbers
    if category == cat_1:   # not
        return 0
    elif category == cat_2:
        return 1
    else:
        return -1

def classify_and_map(dataset, cat_1, cat_2, model, tokenizer):
    """
    Classifies sentences in a dataset and maps the categories to numbers.

    Parameters:
    - dataset: The dataset containing sentences to classify.
    - cat_1, cat_2: The categories for classification.
    - model: The classification model.
    - tokenizer: The tokenizer for the model.

    Returns:
    A list of tuples containing the classification (as a number) and probability for each sentence.
    """

    results = []
    for index, sentence in enumerate(dataset['Sentence']):
        print(f"Processing sentence {index + 1}/{len(dataset)}: {sentence[:50]}...")
        model_output = classify_sentence(sentence, cat_1, cat_2, model, tokenizer)
        category, probability = extract_json(model_output)
        classification = map_category_to_number(category, cat_1, cat_2)
        print(f"Classification result: {classification} with Probability: {probability}")
        results.append((classification, probability))
    return results


def process_and_save_batch(dataset, start_index, batch_size, output_file, dataset_type, results, cat_1, cat_2, model, tokenizer):
    """
    Processes a batch of sentences for classification, saves the results to a CSV file,
    and updates the results list.

    Parameters:
    - dataset: The dataset containing sentences.
    - start_index: The index of the first sentence in the batch.
    - batch_size: The number of sentences in the batch.
    - output_file: The file path to save the results.
    - dataset_type: A label indicating the type of dataset (e.g., 'original', 'perturbed').
    - results: The list to which the batch results will be appended.
    - cat_1, cat_2: The categories for classification.
    - model: The classification model.
    - tokenizer: The tokenizer for the model.

    Returns:
    The runtime in seconds for processing the batch.
    """

    start_time = time.time()

    end_index = min(start_index + batch_size, len(dataset))
    batch = dataset.iloc[start_index:end_index]
    batch_results = classify_and_map(batch, cat_1, cat_2, model, tokenizer)

    # Save classification results and probabilities to the CSV file
    with open(output_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for index, (classification, probability) in enumerate(batch_results, start=start_index):
            writer.writerow([index, classification, probability, dataset_type])

    end_time = time.time()
    runtime_seconds = end_time - start_time

    print(f"Processed batch starting from index {start_index} in {runtime_seconds:.2f} seconds.")
    results.extend(batch_results)
    return runtime_seconds


def initialize_output_file(root_path, safe_dataset_name, perturb_type, perturb_detail):
    """
    Initializes the output file for classification results, creating directories if necessary.

    Parameters:
    - root_path: The base path for saving the output file.
    - safe_dataset_name: A filesystem-safe version of the dataset name.
    - perturb_type: The type of perturbation applied to the dataset.
    - perturb_detail: Specific details about the perturbation.

    Returns:
    The path to the initialized output file.
    """

    output_path = os.path.join(root_path, 'log')

    # if dir does not exist, create the path
    os.makedirs(output_path, exist_ok=True)
    
    output_file = f'{perturb_type}_{safe_dataset_name}_{perturb_detail}_classification_batch_results.csv'
    output_file_path = os.path.join(output_path, output_file)

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Classification", "Probability", "Dataset Type"])

    return output_file_path


