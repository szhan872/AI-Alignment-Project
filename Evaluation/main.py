# Import necessary libraries

import os
import argparse
from data_loader import load_dataset
from model_loader import mistral_model_loading
from classify import initialize_output_file, process_and_save_batch
from fairness_calculate import conf_with_label, conf_without_label, calculate_fairness_score
from export_results import export_result


def main():
    print("start!")
    parser = argparse.ArgumentParser(description="Apply perturbations to a dataset and process it through a machine learning model for fairness evaluation.")

    # Add arguments to the parser for various configurable options
    parser.add_argument("--dataset_name", type=str, default="imdb", help="Name of the dataset.")
    parser.add_argument("--label_class_1", type=str, default="0", help="Class name for the negative sentiment.")
    parser.add_argument("--label_class_2", type=str, default="1", help="Class name for the positive sentiment.")
    parser.add_argument("--perturb_type", type=str, default="gender", choices=['race', 'gender', 'dialect'], help="Type of perturbation to apply.")
    parser.add_argument("--task_type", type=str, default="sentiment analysis", choices=['text classification', 'toxicity detection', 'sentiment analysis'], help="Type of ML task.")
    parser.add_argument("--root_path", type=str, default=os.path.dirname(os.path.abspath(__file__)), help="Root directory for saving data and results.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset within the root directory.")

    parser.add_argument("--model_name", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="Name of the language model to use.")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of samples to process in each batch.")

    args = parser.parse_args()


    # Set dataset_path to a default value if not provided
    if args.dataset_path is None:
        args.dataset_path = os.path.join(args.root_path, 'datasets')

    # Map the perturb_type to a detailed perturbation description
    original_type_list = ['race', 'gender', 'dialect']
    perturb_type_list = ['whiteAmerican_to_blackAmerican', 'male_to_female', 'SAE_to_AAVE']
    choice_number = original_type_list.index(args.perturb_type)
    perturb_detail = perturb_type_list[choice_number]

    # Construct a safe dataset name for file operations
    safe_dataset_name = args.dataset_name.replace('/', '_')

    print("1. model loading ...")
    model, tokenizer = mistral_model_loading(args.model_name)

    print("model loading success.")
    print("2. dataset loading ...")
    original_dataset, perturbed_dataset, original_dataset_path, perturbed_dataset_path = load_dataset(
        safe_dataset_name, args.dataset_path, args.perturb_type, perturb_detail)
    print("loading dataset success.")

    print("3. initializing output file ...")
    output_file = initialize_output_file(
        args.root_path, safe_dataset_name, args.perturb_type, perturb_detail)
    print("initializing output file success.")

    # Initialize lists to store classification results for original and perturbed datasets

    original_dataset_classifications = []
    perturbed_dataset_classifications = []

    num = len(original_dataset) # Adjust the number for testing if needed

    print("4. testing original dataset ...")

    # Process the original dataset in batches
    for start_index in range(0, num, args.batch_size):
        runtime_seconds_original = process_and_save_batch(
            original_dataset, start_index, args.batch_size, output_file, "original", original_dataset_classifications, args.label_class_1, args.label_class_2,
            model, tokenizer)
        print(f"Original batch {start_index} runtime: {runtime_seconds_original:.2f} seconds")
    print("testing original dataset success.")

    print("5. testing perturb dataset ...")

    # Process the perturbed dataset in batches
    for start_index in range(0, num, args.batch_size):
        runtime_seconds_perturb = process_and_save_batch(
            perturbed_dataset, start_index, args.batch_size, output_file, "perturbed", perturbed_dataset_classifications, args.label_class_1, args.label_class_2,
            model, tokenizer)
        print(f"Perturbed batch {start_index} runtime: {runtime_seconds_perturb:.2f} seconds")

    print("testing perturb dataset success.")
    print("Original Dataset Classifications:", original_dataset_classifications)
    print("Perturbed Dataset Classifications:", perturbed_dataset_classifications)


    # Extract classification results for analysis

    original_classifications = [result[0] for result in original_dataset_classifications]
    print("Original Dataset Classifications:", original_classifications)

    perturbed_classifications = [result[0] for result in perturbed_dataset_classifications]
    print("Perturbated Dataset Classifications:", perturbed_classifications)

    print("6. calculating confusion matrix ...")
    # Calculate confusion matrices for before and after perturbation
    TP_before, FP_before, FN_before, TN_before, \
    TP_perturbed, FP_perturbed, FN_perturbed, TN_perturbed = conf_with_label(
                    original_dataset, perturbed_dataset,
                    original_dataset_classifications, perturbed_dataset_classifications)

    TP, FP, FN, TN, chi2, p, dof, expected = conf_without_label(
                    original_dataset, perturbed_dataset,
                    original_dataset_classifications, perturbed_dataset_classifications)

    print("calculating confusion matrix success.")
    print("7. calculating fairness score ...")
    # Calculate and print the fairness score and other disparity metrics
    fairness_score, accuracy_disparity, ppp_disparity, \
    recall_disparity, specificity_disparity, etr_disparity = calculate_fairness_score(
        TP_before, FP_before, FN_before, TN_before,
        TP_perturbed, FP_perturbed, FN_perturbed, TN_perturbed)
    print("calculating fairness score success.")
    print("[jus in case, fairness score:] ", fairness_score)
    print("8. exporting results ...")
    # Export the results to files
    export_result(args.task_type, args.perturb_type, perturb_detail, args.dataset_name,
                  runtime_seconds_original, runtime_seconds_perturb,
                  original_dataset_path, perturbed_dataset_path,
                  args.label_class_1, args.label_class_2,
                  original_dataset_classifications, perturbed_dataset_classifications,
                  TP, FP, FN, TN, chi2, p, dof, expected,
                  TP_before, FP_before, FN_before, TN_before,
                  TP_perturbed, FP_perturbed, FN_perturbed, TN_perturbed,
                  accuracy_disparity, ppp_disparity, recall_disparity, specificity_disparity, etr_disparity,
                  fairness_score,
                  args.root_path, safe_dataset_name)

    print("exporting results success.")


if __name__ == '__main__':
    main()
