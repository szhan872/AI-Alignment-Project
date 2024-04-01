import os
import csv
import re
from datasets import load_dataset
from random import Random
from typing import Optional
import os
from data_loader import load_white_american_first_names, load_dialect_mapping, contains_and_highlight_terms
from datasets.utils.py_utils import NonMutableDict

white_american_first_names = load_white_american_first_names()
dialect_map_keys = load_dialect_mapping()

# 术语映射
terms_maps = {
    'gender': [
        "son", "sons", "boy", "boys", "man", "men", "father", "fathers", "brother", "brothers",
        "nephew", "nephews", "king", "kings", "waiter", "waiters", "dad", "dads", "stepson",
        "stepsons", "stepfather", "stepfathers", "grandson", "grandsons", "grandfather",
        "grandfathers", "male", "males", "he", "him", "his", "himself"
    ],
    # 为种族和方言添加示例术语，这里仅作为示意
    'race': white_american_first_names,
    'dialect': dialect_map_keys
}

def extract_and_write_terms(dataset, safe_dataset_name, original_type, perturb_type, max_count=100,
                            task_type='text classification', filter_sentence_flag=True, balance_label_flag=True,
                            label_cat_1=None, label_cat_2=NonMutableDict,
                            root_path: Optional[str] = None,
                            dataset_path: Optional[str] = None):
    print('task_type:', task_type)
    print('original_type:', original_type)
    print('perturb_type:', perturb_type)
    print('filter_sentence_flag:', filter_sentence_flag)
    print('balance_label_flag:', balance_label_flag)

    if root_path is None:
        root_path = os.path.dirname(os.path.abspath(__file__))
    # 如果dataset_path没有被明确提供，设置为root_path下的'datasets'目录
    if dataset_path is None:
        dataset_path = os.path.join(root_path, 'datasets')

    # 确保目录存在
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    directory_path = dataset_path
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # 数据集的实际大小可能会影响我们的平衡标签逻辑
    dataset_size = len(dataset)
    if dataset_size < max_count:
        print(
            f"Dataset size ({dataset_size}) is smaller than max_count ({max_count}). Adjusting max_count accordingly.")
        max_count = dataset_size  # 调整max_count为数据集的大小

    # 如果需要平衡标签，调整每个标签应有的最大数量
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
            break  # 如果每个标签的计数都达到了max_per_label，则停止

        # 根据任务类型确定标签处理方式
        if task_type == 'toxicity detection':
            label = 1 if example['Label'] > 0.5 else 0
            # print(f"toxicity: {example['Label']}, label: {label}")
        else:
            # 假设这里的label_name变量已经根据具体的数据集进行了设置
            label_cat_1 = str(label_cat_1)
            label_cat_2 = str(label_cat_2)

            label = str(example['Label'])

            # print(f"label: {label} (type: {type(label)})")
            # print(f"label_cat_1: {label_cat_1} (type: {type(label_cat_1)})")
            # print(f"label_cat_2: {label_cat_2} (type: {type(label_cat_2)})")

            
            if label == label_cat_1:
                print("bingo")
                label = 0
            elif label == label_cat_2:
                # print("[label] = ", label, "[label_cat_1] = ", label_cat_1, "[label_cat_2] = ", label_cat_2)
                label = 1
                # print ("[decesion] = ", label)
            else:
                print(f"Unknown label: {label}")
                continue  # 或者选择其他逻辑

        highlighted_text = example['Sentence']  # 默认高亮文本就是原文

        # 如果设置了仅筛选含有特定术语的句子
        if filter_sentence_flag:
            found, _ = contains_and_highlight_terms(example['Sentence'], terms)
            if not found:
                continue  # 如果未找到术语，则跳过当前例子

        # 添加句子到collected_sentences中，并更新true_count或false_count
        if not balance_label_flag or (label == 1 and true_count < max_per_label) or (
                label == 0 and false_count < max_per_label):
            collected_sentences.append((highlighted_text, label))
            if label == 1:
                true_count += 1
            else:
                false_count += 1
        print(f"true_count: {true_count}, false_count: {false_count}")

    # 保存至CSV
    csv_file_path = f'{directory_path}/original_{original_type}_{safe_dataset_name}_{perturb_type}.csv'
    print(csv_file_path)
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Sentence', 'Label'])
        for sentence, label in collected_sentences:
            writer.writerow([sentence, label])

    print(f"Total lines included: {len(collected_sentences)}")

