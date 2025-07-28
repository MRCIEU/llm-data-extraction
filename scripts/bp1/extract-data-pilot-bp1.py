from pathlib import Path
import json

import torch
from environs import env

from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig

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
quantization_config = QuantoConfig(weights="int8")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    token=access_token,
    quantization_config=quantization_config,
)
print("Loaded model")

metadataprompt = {
    "role": "user",
    "content": """What are the exposures, outcomes in this abstract? If there are multiple exposures or outcomes, provide them all. If there are no exposures or outcomes, provide an empty list. Also categorize the exposures and outcomes into the following groups using the exact category names provided: 
- molecular
- socioeconomic
- environmental
- behavioural
- anthropometric
- clinical measures
- infectious disease
- neoplasm
- disease of the blood and blood-forming organs
- metabolic disease
- mental disorder
- disease of the nervous system
- disease of the eye and adnexa
- disease of the ear and mastoid process
- disease of the circulatory system
- disease of the digestive system
- disease of the skin and subcutaneous tissue
- disease of the musculoskeletal system and connective tissue
- disease of the genitourinary system
If an exposure or outcome does not fit into any of these groups, specify "Other". 

List the analytical methods used in the abstract. Match the methods to the following list of exact method names. If a method is used that is not in the list, specify "Other" and also provide the name of the method. The list of methods is as follows:
- two-sample mendelian randomization
- multivariable mendelian randomization
- colocalization
- network mendelian randomization
- triangulation
- reverse mendelian randomization
- one-sample mendelian randomization
- negative controls
- sensitivity analysis
- non-linear mendelian randomization
- within-family mendelian randomization

Provide a description of the population(s) on which the study described in the abstract was based.

Provide your answer in strict pretty JSON format using exactly the format as the example output and without markdown code blocks. Any error messages and explanations must be included in the JSON output with the key "metainformation".
""",
}

resultsprompt = {
    "role": "user",
    "content": """
List all of the results in the abstract, with each entry comprising: exposure, outcome, beta, units, odds ratio, hazard ratio, 95% confidence interval, standard error, and P-value. If any of these fields is missing, substitute them with "null". Add a field called "direction" which describes whether the exposure "increases" or "decreases" the outcome. 
Provide your answer in strict pretty JSON format using exactly the format as the example output and without markdown code blocks. You must only include values explicitly written in the abstract. Any error messages and explanations must be included in the JSON output with the key "resultsinformation". 

""",
}


metadataexample = {
    "role": "assistant",
    "content": """This is an example output in JSON format: 
    { "metadata": {
    "exposures": [
    {
        "id": "1",
        "trait": "Particulate matter 2.5",
        "category": "Environmental"
    },
    {
        "id": "2",
        "trait": "Type 2 diabetes",
        "category": "metabolic disease"
    },
    {
        "id": "3",
        "trait": "Body mass index",
        "category": "Anthropometric"
    }
    ],
    "outcomes": [
    {
        "id": "1",
        "trait": "Forced expiratory volume in 1 s",
        "category": "Clinical measure"
    },
    {
        "id": "2",
        "trait": "Forced vital capacity",
        "category": "Clinical measure"
    },
    {
        "id": "3",
        "trait": "Gastroesophageal reflux disease",
        "category": "disease of the digestive system"
    },
    {
        "id": "4",
        "trait": "Non-alcoholic fatty liver disease (NAFLD)",
        "category": "disease of the digestive system"
    }
    ],
    "methods": ["two-sample mendelian randomization", "multivariable mendelian randomization", "colocalisation", "network mendelian randomization"],
    "population": ["European men", "Breast cancer patients", "African-Americans"],
    "metainformation": {
        "error": "No information on population is provided in abstract",
        "explanation": "Some methods do not match those listed in the prompt"
    }
    }
    }
    """,
}

resultsexample = {
    "role": "assistant",
    "content": """This is an example output in JSON format: 
    {
    "results": [
        {
            "exposure": "Particulate matter 2.5"},
            "outcome": "Forced expiratory volume in 1 s"
            "beta": 0.154,
            "units": "mmHg",
            "hazard ratio": null,
            "odds ratio": null,
            "95% CI": [0.101,0.215],
            "SE": 0.102,
            "P-value": 0.0015,
            "Direction": "increases"
        },
        {
            "exposure": "Body mass index"},
            "outcome": "Gastroesophageal reflux disease"
            "beta": null,
            "units": null,
            "hazard ratio": null,
            "odds ratio": 1.114,
            "95% CI": [1.021,1.314],
            "SE": null,
            "P-value": 0.0157,
            "Direction": "increases"
        },
        {
            "exposure": "Body mass index"},
            "outcome": "Non-alcoholic fatty liver disease (NAFLD)"
            "beta": null,
            "units": null,
            "hazard ratio": null,
            "odds ratio": null,
            "95% CI": [null,null],
            "SE": null,
            "P-value": null,
            "Direction": "increases"
        }
    ]
    "resultsinformation": {
        "error": "No results provided in abstract",
        "explanation": "P-values were string, not numeric values"
    }
    }
    """,
}


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
                metadataexample,
                metadataprompt,
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
                resultsexample,
                resultsprompt,
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
