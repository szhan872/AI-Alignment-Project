# Bias Detection Model

This directory contains the bias detection model that is trained for classifying textual data into 1 for "biased" and 0 for "non-biased" binary category. Overall, this model has better performance than other models (outperform Dbias on unseen datasets such as MBIC headlines: news_headlines_usa_biased.csv and news_headlines_usa_neutral.csv). It also runs 10 more times faster than Dbias and HELM metric. Both of these algorithms are implemented in src/HELM+Dbias folder.

### Prerequisites
- Python 3.9+
- recommended: pytorch with cuda:

```python
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Installation
1. Clone the repository

2. Install required packages: pip install -r requirements.txt

## Model
- **Bias Detection Model:** The link to the model is D1V1DE/bias-detection on huggingface. A DistillRoBERTa bias classification model fine-tuned from valurank/distilroberta-bias. MBIC dataset is used as training data.

- See Hugging Face webpage for more information: https://huggingface.co/D1V1DE/bias-detection

## HELM and Dbias
These are also bias evaluation methods that are tested in early stage of this project. These two methods are then proved to be slow and inaccurate for this task. For installation and testing, please install the provided requirement.txt file in that folder again.

## Contact
For questions or feedback, please open an issue in the repository or contact us directly at `ldvdzhang@gmail.com`.

Thank you for exploring our AI Alignment Model!

