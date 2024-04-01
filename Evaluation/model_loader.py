# Necessary imports for handling arrays, operating system interactions, and subprocess management

import numpy
import os
import subprocess
import sys

# Set environment variables required for running the model, especially important in environments like Colab
os.environ['LC_ALL'] = "en_US.UTF-8"
os.environ['LD_LIBRARY_PATH'] = "/usr/lib64-nvidia"
os.environ['LIBRARY_PATH'] = "/usr/local/cuda/lib64/stubs"

# Update dynamic linker run-time bindings
subprocess.run(['ldconfig', '/usr/lib64-nvidia'])

# Clone the repository if it doesn't already exist, to load the offloading model components
if not os.path.exists('mixtral-offloading'):
    subprocess.run(['git', 'clone', 'https://github.com/dvmazur/mixtral-offloading.git'], stdout=subprocess.DEVNULL)


# Change directory to the cloned repository and install required Python dependencies
os.chdir('mixtral-offloading')
subprocess.run(['pip', 'install', '-r', 'requirements.txt'], stdout=subprocess.DEVNULL)



model_dir = os.path.join(os.getcwd(), 'Mixtral-8x7B-Instruct-v0.1-offloading-demo')
if not os.path.exists(model_dir):
    print("0. download model ...")
    # 定义模型名称和下载目录
    model_name = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
    local_dir = "Mixtral-8x7B-Instruct-v0.1-offloading-demo"

    # 构建 huggingface-cli 命令
    command = ["huggingface-cli", "download", model_name, "--quiet", "--local-dir", local_dir]

    # 执行命令
    subprocess.run(command, check=True)

sys.path.append(os.path.join(os.getcwd(), "mixtral-offloading", "src"))
sys.path.append("mixtral-offloading")

# Revert back to the original directory if necessary
os.chdir('..')


script_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(script_dir, 'mixtral-offloading', 'Mixtral-8x7B-Instruct-v0.1-offloading-demo', 'model.safetensors.index.json')
print("path:", json_file_path)
print("current path:", os.getcwd())


import torch
from torch.nn import functional as F
from hqq.core.quantize import BaseQuantizeConfig
from huggingface_hub import snapshot_download
from IPython.display import clear_output
from tqdm.auto import trange
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import logging as hf_logging
from src.build_model import OffloadConfig, QuantConfig, build_model


# Define a function to load the Mistral model along with its tokenizer
def mistral_model_loading(model_name):
    """
    Loads a quantized and offloaded Mistral model along with its tokenizer.

    This function sets up the environment for running the Mistral model by adjusting
    environment variables, configuring offloading and quantization parameters to manage
    GPU memory usage, and then building the model with these configurations. The function
    also ensures that the model's tokenizer is loaded, allowing for input text to be
    properly formatted before being fed into the model.

    Parameters:
    - model_name: The name of the model to load. This is used to identify the correct
      tokenizer that matches the model.

    Returns:
    - model: The loaded model, configured for quantization and offloading, ready for
      inference.
    - tokenizer: The tokenizer corresponding to the loaded model, used for processing
      text inputs.

    Note:
    The function assumes that the required repositories and dependencies have already
    been cloned and installed, respectively. It also sets certain environment variables
    that are necessary for the model to run correctly, especially in environments like
    Google Colab.
    """

    print("Model loading initiated...")
    model_name = model_name
    quantized_model_name = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
    state_path = "mixtral-offloading/Mixtral-8x7B-Instruct-v0.1-offloading-demo"

    # Load the model configuration from Hugging Face's model hub
    config = AutoConfig.from_pretrained(quantized_model_name)

    # Set the computing device
    device = torch.device("cuda:0")

    # Configure offloading to manage GPU memory usage efficiently
    ##### Change this to 5 if you have only 12 GB of GPU VRAM #####
    offload_per_layer = 4
    # offload_per_layer = 5
    ###############################################################

    num_experts = config.num_local_experts
    offload_config = OffloadConfig(
        main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
        offload_size=config.num_hidden_layers * offload_per_layer,
        buffer_size=4,
        offload_per_layer=offload_per_layer,
    )

    # Configure quantization for attention and feed-forward networks to reduce model size
    attn_config = BaseQuantizeConfig(
        nbits=4,
        group_size=64,
        quant_zero=True,
        quant_scale=True,
    )
    attn_config["scale_quant_params"]["group_size"] = 256

    ffn_config = BaseQuantizeConfig(
        nbits=2,
        group_size=16,
        quant_zero=True,
        quant_scale=True,
    )
    quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)

    print("current path:", os.getcwd())
    # Build the model with the specified configurations
    model = build_model(
        device=device,
        quant_config=quant_config,
        offload_config=offload_config,
        state_path=state_path,
    )

    # Load the tokenizer corresponding to the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Return both the model and its tokenizer
    return model, tokenizer