import argparse
import os
from random import Random

from data_loader import load_dataset_func
from original_dataset import extract_and_write_terms
from perturb_dataset import apply_perturbation_and_save



def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Apply perturbations to a dataset and process it.")

    # Define command-line arguments for the program
    parser.add_argument("--dataset_name", type=str, default="imdb", help="Name of the dataset.")
    parser.add_argument("--dataset_sub_name", type=str, default='', help="Sub-name of the dataset, if applicable.")
    parser.add_argument("--sentence_name", type=str, default="Sentence", help="Column name for text data, usually the first column.")
    parser.add_argument("--label_name", type=str, default="Label", help="Column name for class labels.")
    parser.add_argument("--label_class_1", type=str, default="0", help="Class name/label for the negative class in the dataset.")
    parser.add_argument("--label_class_2", type=str, default="1", help="Class name/label for the positive class in the dataset.")
    parser.add_argument("--perturb_type", type=str, default="gender", choices=['race', 'gender', 'dialect'], help="Type of perturbation to apply.")
    parser.add_argument("--task_type", type=str, default="sentiment analysis", choices=['text classification', 'toxicity detection', 'sentiment analysis'], help="Task type for the dataset.")
    parser.add_argument("--enable_filter", action='store_true', help="Filter sentences to only those containing perturbation-specific words. Defaults to including all sentences.")
    parser.add_argument("--enable_balance", action='store_true', help="Balance labels to have an equal number of instances for each label. Defaults to using the dataset as-is.")
    parser.add_argument("--dataset_size", type=int, default=100, help="Maximum number of sentences to include in the processed dataset.")
    parser.add_argument("--root_path", type=str, default=os.path.dirname(os.path.abspath(__file__)), help="Root directory path for saving data and results.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Specific path within the root path where the dataset is located.")

    args = parser.parse_args()


    # Set the dataset path based on root_path if not provided
    if args.dataset_path is None:
        args.dataset_path = os.path.join(args.root_path, 'datasets')

    # Determine perturbation detail based on the selected perturbation type
    original_type_list = ['race', 'gender', 'dialect']
    perturb_type_list = ['whiteAmerican_to_blackAmerican', 'male_to_female', 'SAE_to_AAVE']
    choice_number = original_type_list.index(args.perturb_type)
    perturb_type = perturb_type_list[choice_number]

    # Construct a filesystem-safe dataset name for file naming
    safe_dataset_name = args.dataset_name.replace('/', '_')

    # Initialize random number generator
    rng = Random()

    # Load the dataset
    dataset = load_dataset_func(args.dataset_name, args.dataset_sub_name,
                                [args.sentence_name, args.label_name],
                                {args.sentence_name: 'Sentence', args.label_name: 'Label'})

    # Extract terms and write initial data
    extract_and_write_terms(dataset, safe_dataset_name,
                            original_type=args.perturb_type,
                            perturb_type=perturb_type,
                            max_count=args.dataset_size,
                            task_type=args.task_type,
                            filter_sentence_flag=args.enable_filter,
                            balance_label_flag=args.enable_balance,
                            label_cat_1=args.label_class_1,
                            label_cat_2=args.label_class_2,
                            root_path=args.root_path,
                            dataset_path=args.dataset_path)

    # Apply perturbation to the dataset and save the result
    apply_perturbation_and_save(args.root_path, args.dataset_path, choice_number,
                                args.perturb_type, perturb_type, safe_dataset_name)


if __name__ == '__main__':
    main()

