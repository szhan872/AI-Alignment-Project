# import statements
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
# from tqdm import tqdm
# from tqdm.auto import tqdm
# tqdm.pandas()

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("D1V1DE/bias-detection")
model = AutoModelForSequenceClassification.from_pretrained("D1V1DE/bias-detection")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the classification function
def classify_text(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    ## Move the tokenized inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.to(device)
    
    # Forward pass through the model
    outputs = model(**inputs)
    
    # Get the predicted label
    predicted_label = outputs.logits.argmax().item()
    
    return predicted_label