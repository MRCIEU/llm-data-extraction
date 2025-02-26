from pathlib import Path
import json

import torch
from environs import env

# from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

from local_funcs import prompts

# Configure params
env.read_env()
startpoint = 0
endpoint = 101
output_path = Path(".") / "output"
access_token = env("HUGGINGFACE_TOKEN")


# Specify model
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "cuda"
dtype = torch.bfloat16

# Get abstracts
data_path = Path(".") / "data"
path_to_pubmed = data_path / "mr-pubmed-abstracts" / "data" / "pubmed.json"
assert path_to_pubmed.exists(), print("pubmed.json not found")
with path_to_pubmed.open("r") as f:
    pubmed = json.load(f)

print("Loaded abstracts")

# Set up quantized model
tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
# quantization_config = QuantoConfig(weights="int4")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    token=access_token,
    # load_in_8bit=True,
)
print("Loaded model")


def respond(prompt):
    """Instruct the model, returning a response
    Parameters:
        prompt: a string containing a LLM prompt
    Returns: the LLM response
    """
    input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=1000, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


def clean_result(result):
    """Clean the results, removing anything outside the JSON
    Parameters:
        result: Output from the LLM
    Returns: cleaned JSON output
    """
    result = result.split("}")
    output = json.loads("}".join(result[0:-1]) + "}")
    return output


def extract(messages):
    """Instruct the model, returning a response
    Parameters:
        messages: a string containing an LLM prompt
    Returns: the LLM response
    """
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        # top_p=0.15,
    )
    response = outputs[0][input_ids.shape[-1] :]
    return tokenizer.decode(response, skip_special_tokens=True)


fulldata = []

# Loop over all specified abstracts in the dataset
for abstract in pubmed[startpoint:endpoint]:
    try:
        completion = extract(
            [
                {
                    "role": "system",
                    "content": "You are a data scientist responsible for extracting accurate information from research papers. You answer each question with a single JSON string.",
                },
                {
                    "role": "user",
                    "content": f"""
             This is an abstract from a Mendelian randomization study. 
                "{abstract["ab"]}"   """,
                },
                prompts.metadataexample,
                prompts.metadataprompt,
            ]
        )
        completion2 = extract(
            [
                {
                    "role": "system",
                    "content": "You are a data scientist responsible for extracting accurate information from research papers. You answer each question with a single JSON string.",
                },
                {
                    "role": "user",
                    "content": f"""
             This is an abstract from a Mendelian randomization study. 
                "{abstract["ab"]}"   """,
                },
                prompts.resultsexample,
                prompts.resultsprompt,
            ]
        )
        result1 = clean_result(completion)
        result2 = clean_result(completion2)
        output = dict(abstract, **result1, **result2)
        fulldata.append(output)
    except Exception as e:
        print(f"""\n\n=========== {abstract["pmid"]} ==========""")
        print("""\n=========== FAILED! ==========""")
        # print(abstract)
        print(e)
        result1 = {"metadata": {}, "metainformation": {"error": f"Failed {e}"}}
        result2 = {"results": {}, "resultsinformation": {"error": f"Failed {e}"}}
        output = dict(abstract, **result1, **result2)
        fulldata.append(output)

print(json.dumps(fulldata, indent=4))

out_file = output_path / "mr_extract_llama3_sample_0.json"
with out_file.open("w") as f:
    json.dump(fulldata, f, indent=4)
