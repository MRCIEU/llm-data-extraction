import json
from pprint import pprint

import torch
import transformers
from environs import env
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig, AutoProcessor, Llama4ForConditionalGeneration

from local_funcs import chat_funcs, prompt_funcs
from yiutils.project_utils import find_project_root

proj_root = find_project_root("justfile")
data_dir = proj_root / "data"

print(transformers.__version__)
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

env.read_env(proj_root / ".env")
access_token = env("HUGGINGFACE_TOKEN")

path_to_mr_pubmed_data = (
    data_dir / "intermediate" / "mr-pubmed-data" / "mr-pubmed-data.json"
)
assert path_to_mr_pubmed_data.exists(), (
    f"Data file {path_to_mr_pubmed_data} does not exist."
)

with open(path_to_mr_pubmed_data, "r") as f:
    mr_pubmed_data = json.load(f)

article_data = mr_pubmed_data[0]
