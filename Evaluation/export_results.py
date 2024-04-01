import numpy as np
import json
import datetime
import os



def default_converter(obj):
    """
    Convert numpy objects to a JSON serializable format.

    Parameters:
    - obj: The object to convert.

    Returns:
    A JSON serializable format of the input object.

    Raises:
    TypeError if the object cannot be serialized to JSON.
    """

    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type '{obj.__class__.__name__}' is not JSON serializable")


def export_result(task_type, perturb_type, perturb_detail, dataset_name,
                  runtime_seconds_original, runtime_seconds_perturb,
                  original_dataset_path, perturbed_dataset_path,
                  label_cat_1, label_cat_2,
                  original_dataset_classifications, perturbed_dataset_classifications,
                  TP, FP, FN, TN, chi2, p, dof, expected,
                  TP_before, FP_before, FN_before, TN_before,
                  TP_perturbed, FP_perturbed, FN_perturbed, TN_perturbed,
                  accuracy_disparity, ppp_disparity, recall_disparity, specificity_disparity, etr_disparity,
                  fairness_score,
                  root_path, safe_dataset_name
                  ):
    """
    Export the analysis results to a JSON file, including confusion matrix components, disparity measures,
    and fairness score among other metadata.

    Parameters:
    - Multiple parameters capturing the details of the analysis, including task details, dataset paths,
      classification results, confusion matrix components, disparity measures, and the fairness score.
    - root_path, safe_dataset_name: Base directory for saving the output and a safe version of the dataset name.

    The function saves a comprehensive record of the analysis to a JSON file.
    """

    # Format the current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Compile all relevant information into a record dictionary
    record = {
        'task': task_type,
        'perturb_type': perturb_type,
        'perturb_detail': perturb_detail,
        'dataset_name': dataset_name,
        'date': current_date,
        'runtime_seconds_original': runtime_seconds_original,
        'runtime_seconds_perturb': runtime_seconds_perturb,
        'original_dataset_path': original_dataset_path,
        'perturbed_dataset_path': perturbed_dataset_path,
        'categories': [label_cat_1, label_cat_2],
        'original_dataset_classifications': original_dataset_classifications,
        'perturbed_dataset_classifications': perturbed_dataset_classifications,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'chi2': chi2,
        'p': p,
        'dof:': dof,
        'original_TP': TP_before,
        'original_FP': FP_before,
        'original_FN': FN_before,
        'original_TN': TN_before,
        'perturbed_TP': TP_perturbed,
        'perturbed_FP': FP_perturbed,
        'perturbed_FN': FN_perturbed,
        'perturbed_TN': TN_perturbed,
        'accuracy_disparity': accuracy_disparity,
        'positive_prediction_proportion_disparity': ppp_disparity,
        'recall_disparity': recall_disparity,
        'specificity_disparity': specificity_disparity,
        'error_type_ratio_disparity': etr_disparity,
        'fairness_score': fairness_score
    }

    # Define the output path and file for saving the record
    output_path = os.path.join(root_path, 'log')
    os.makedirs(output_path, exist_ok=True)  # Ensure the directory exists
    output_file = f'dataset_record_{perturb_type}_{safe_dataset_name}_{perturb_detail}.json'
    output_file = os.path.join(output_path, output_file)

    # Write the record dictionary to a JSON file
    with open(output_file, 'w') as f:
        json.dump(record, f, indent=4, default=default_converter)

    print(f'Dataset record saved to {output_file}')
    print(f'Record: {record}')

