import sklearn
import numpy
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency


def conf_with_label(original_dataset, perturbed_dataset,
                    original_dataset_classifications, perturbed_dataset_classifications):
    """
    Calculate confusion matrices for the original and perturbed datasets based on actual labels.

    Parameters:
    - original_dataset, perturbed_dataset: Datasets containing the true labels.
    - original_dataset_classifications, perturbed_dataset_classifications: Lists of tuples containing classification results and probabilities.

    Returns:
    Tuple of true positive, false positive, false negative, and true negative counts for both original and perturbed datasets.
    """

    num = len(original_dataset)
    # Extract true labels for both datasets
    true_labels_original = original_dataset['Label'][:num]
    true_labels_perturbed = perturbed_dataset['Label'][:num]

    # Filter out invalid classification results
    valid_indices_original = [i for i, x in enumerate(original_dataset_classifications) if x[0] != -1]
    valid_indices_perturbed = [i for i, x in enumerate(perturbed_dataset_classifications) if x[0] != -1]

    # Extract valid true labels and predictions
    valid_true_labels_original = [true_labels_original[i] for i in valid_indices_original]
    valid_predictions_original = [original_dataset_classifications[i][0] for i in valid_indices_original]

    valid_true_labels_perturbed = [true_labels_perturbed[i] for i in valid_indices_perturbed]
    valid_predictions_perturbed = [perturbed_dataset_classifications[i][0] for i in valid_indices_perturbed]

    # Compute confusion matrices for filtered results
    conf_matrix_original = confusion_matrix(valid_true_labels_original,
                                            valid_predictions_original)
    TP_before = conf_matrix_original[1, 1]
    FP_before = conf_matrix_original[0, 1]
    FN_before = conf_matrix_original[1, 0]
    TN_before = conf_matrix_original[0, 0]

    conf_matrix_perturbated = confusion_matrix(valid_true_labels_perturbed, valid_predictions_perturbed)

    TP_perturbed = conf_matrix_perturbated[1, 1]
    FP_perturbed = conf_matrix_perturbated[0, 1]
    FN_perturbed = conf_matrix_perturbated[1, 0]
    TN_perturbed = conf_matrix_perturbated[0, 0]

    # Output confusion matrix components
    print("Original Dataset - TP:", TP_before, "FP:", FP_before, "FN:", FN_before, "TN:", TN_before)
    print("Perturbated Dataset - TP:", TP_perturbed, "FP:", FP_perturbed, "FN:", FN_perturbed, "TN:", TN_perturbed)

    return TP_before, FP_before, FN_before, TN_before, TP_perturbed, FP_perturbed, FN_perturbed, TN_perturbed


def conf_without_label(original_dataset, perturbed_dataset,
                    original_dataset_classifications, perturbed_dataset_classifications):
    """
    Calculate confusion matrices without actual labels, comparing original and perturbed datasets' predictions.

    Parameters:
    - original_dataset, perturbed_dataset: Datasets for which classifications were made.
    - original_dataset_classifications, perturbed_dataset_classifications: Lists of tuples containing classification results and probabilities.

    Returns:
    Tuple of confusion matrix components (TP, FP, FN, TN) and chi-square test results.
    """

    num = len(original_dataset)

    # Filter out invalid classification results
    valid_indices_original = [i for i, x in enumerate(original_dataset_classifications) if x[0] != -1]
    valid_indices_perturbed = [i for i, x in enumerate(perturbed_dataset_classifications) if x[0] != -1]

    valid_predictions_original = [original_dataset_classifications[i][0] for i in valid_indices_original]
    valid_predictions_perturbed = [perturbed_dataset_classifications[i][0] for i in valid_indices_perturbed]

    # Extract valid predictions for common valid indices
    common_valid_indices = [i for i in range(num) if
                            original_dataset_classifications[i][0] != -1 and perturbed_dataset_classifications[i][
                                0] != -1]
    common_valid_predictions_original = [original_dataset_classifications[i][0] for i in common_valid_indices]
    common_valid_predictions_perturbated = [perturbed_dataset_classifications[i][0] for i in common_valid_indices]

    # Compute confusion matrix for valid predictions
    conf_matrix_original = confusion_matrix(common_valid_predictions_original, common_valid_predictions_perturbated)

    TP = conf_matrix_original[1, 1]
    FP = conf_matrix_original[0, 1]
    FN = conf_matrix_original[1, 0]
    TN = conf_matrix_original[0, 0]



    # Perform chi-square test to compare the results
    chi2, p, dof, expected = chi2_contingency(conf_matrix_original)

    # Output test results
    print("TP:", TP, "FP:", FP, "FN:", FN, "TN:", TN)
    print(chi2, p, dof, expected)

    return TP, FP, FN, TN, chi2, p, dof, expected


def calculate_accuracy(TP, FP, FN, TN):
    """Calculate accuracy from confusion matrix components."""
    return (TP + TN) / (TP + FP + FN + TN)


def calculate_positive_prediction_proportion(TP, FP, FN, TN):
    """Calculate the proportion of positive predictions (positive predictive value)."""
    return (TP + FP) / (TP + FP + FN + TN)


def calculate_recall(TP, FN):
    """Calculate recall (true positive rate) from confusion matrix components."""
    return TP / (TP + FN)


def calculate_specificity(TN, FP):
    """Calculate specificity (true negative rate) from confusion matrix components."""
    return TN / (TN + FP)


def calculate_error_type_ratio(FP, FN):
    """Calculate the ratio of false positives to false negatives, handling division by zero."""
    return FP / FN if FN != 0 else float('inf')


def calculate_fairness_score(
        TP_before, FP_before, FN_before, TN_before,
        TP_perturbed, FP_perturbed, FN_perturbed, TN_perturbed):
    """
    Calculate a fairness score based on disparities in accuracy, positive prediction proportion,
    recall, specificity, and error type ratio between original and perturbed datasets.

    Returns:
    Weighted sum of disparities, along with individual disparity measures.
    """

    # Calculate disparities between original and perturbed datasets for various metrics
    accuracy_before = calculate_accuracy(TP_before, FP_before, FN_before, TN_before)
    accuracy_perturbed = calculate_accuracy(TP_perturbed, FP_perturbed, FN_perturbed, TN_perturbed)
    accuracy_disparity = accuracy_before - accuracy_perturbed

    ppp_before = calculate_positive_prediction_proportion(TP_before, FP_before, FN_before, TN_before)
    ppp_perturbed = calculate_positive_prediction_proportion(TP_perturbed, FP_perturbed, FN_perturbed, TN_perturbed)
    ppp_disparity = ppp_before - ppp_perturbed

    recall_before = calculate_recall(TP_before, FN_before)
    recall_perturbed = calculate_recall(TP_perturbed, FN_perturbed)
    recall_disparity = recall_before - recall_perturbed

    specificity_before = calculate_specificity(TN_before, FP_before)
    specificity_perturbed = calculate_specificity(TN_perturbed, FP_perturbed)
    specificity_disparity = specificity_before - specificity_perturbed

    etr_before = calculate_error_type_ratio(FP_before, FN_before)
    etr_perturbed = calculate_error_type_ratio(FP_perturbed, FN_perturbed)
    etr_disparity = etr_before - etr_perturbed


    # Define weights for each disparity measure
    weights = {
        'accuracy_disparity': 0.2,  # weight for accuracy disparity
        'ppp_disparity': 0.2,  # weight for positive prediction proportion disparity
        'recall_disparity': 0.2,  # weight for recall disparity
        'specificity_disparity': 0.2,  # weight for specificity disparity
        'etr_disparity': 0.2  # weight for error type ratio disparity
    }

    # Calculate the weighted sum of disparities as a fairness score
    weighted_sum = (weights['accuracy_disparity'] * accuracy_disparity +
                    weights['ppp_disparity'] * ppp_disparity +
                    weights['recall_disparity'] * recall_disparity +
                    weights['specificity_disparity'] * specificity_disparity +
                    weights['etr_disparity'] * (
                        etr_disparity if etr_disparity != float('inf') else 1))  # 如果ETR是无穷大，使用极值1


    # Print disparity measures
    print("accuracy disparity: ", accuracy_disparity)
    print("ppp disparity: ", ppp_disparity)
    print("recall disparity: ", recall_disparity)
    print("specificity disparity: ", specificity_disparity)
    print("etr disparity: ", etr_disparity)
    print('Fairness Score:', weighted_sum)

    return weighted_sum, accuracy_disparity, ppp_disparity, recall_disparity, specificity_disparity, etr_disparity