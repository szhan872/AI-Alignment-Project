import argparse
import os
from random import Random

from data_loader import load_dataset_func
from original_dataset import extract_and_write_terms
from perturb_dataset import apply_perturbation_and_save



def main():

    parser = argparse.ArgumentParser(description="Apply perturbations to a dataset and process it.")

    # 定义命令行接受的参数
    parser.add_argument("--dataset_name", type=str, default="imdb", help="Name of the dataset.")
    parser.add_argument("--dataset_sub_name", type=str, default='', help="Sub-name of the dataset, if applicable.")
    parser.add_argument("--sentence_name", type=str, default="Sentence", help="the name of your column that involves text, typically the first column.")
    parser.add_argument("--label_name", type=str, default="Label", help="the name of your column that involves classes.")
    parser.add_argument("--label_class_1", type=str, default="0", help="the class name of the negative class of the dataset.")
    parser.add_argument("--label_class_2", type=str, default="1", help="the class name of the positive class of the dataset.")
    parser.add_argument("--perturb_type", type=str, default="gender", choices=['race', 'gender', 'dialect'],
                        help="Original type of perturbation.")
    parser.add_argument("--task_type", type=str, default="sentiment analysis",
                        choices=['text classification', 'toxicity detection', 'sentiment analysis'],
                        help="The task type.")
    parser.add_argument("--enable_filter", action='store_true',
                        help="Enable sentence filtering to only include sentences containing perturb words. Default behavior is to include all sentences.")

    parser.add_argument("--enable_balance", action='store_true',
                        help="Enable label balancing to ensure an equal number of instances for each label. Default behavior is to use dataset as is, without balancing.")

    parser.add_argument("--dataset_size", type=int, default=100, help="Specifies the maximum number of sentences to include in the processed and exported dataset.")
    parser.add_argument("--root_path", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                        help="Root directory path where data and results are saved.")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Specific path under root path where dataset is located.")

    args = parser.parse_args()

    # 如果没有提供 dataset_path，根据 root_path 设置它的值
    if args.dataset_path is None:
        args.dataset_path = os.path.join(args.root_path, 'datasets')

    # 根据original_type确定perturb_type
    original_type_list = ['race', 'gender', 'dialect']
    perturb_type_list = ['whiteAmerican_to_blackAmerican', 'male_to_female', 'SAE_to_AAVE']
    choice_number = original_type_list.index(args.perturb_type)
    perturb_type = perturb_type_list[choice_number]

    # 安全地构造数据集名称，用于文件命名
    safe_dataset_name = args.dataset_name.replace('/', '_')

    rng = Random()

    # 加载数据集
    dataset = load_dataset_func(args.dataset_name, args.dataset_sub_name,
                                [args.sentence_name, args.label_name],
                                {args.sentence_name: 'Sentence', args.label_name: 'Label'})

    # 提取并写入数据
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

    # 应用扰动并保存
    apply_perturbation_and_save(args.root_path, args.dataset_path, choice_number,
                                args.perturb_type, perturb_type, safe_dataset_name)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

