import json

from transformers import AutoModelForCausalLM, AutoTokenizer


def respond(
    prompt: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, device: str
):
    """Instruct the model, returning a response
    Parameters:
        prompt: a string containing a LLM prompt
    Returns: the LLM response
    """
    input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=1000, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


def clean_result(result: str):
    """Clean the results, removing anything outside the JSON
    Parameters:
        result: Output from the LLM
    Returns: cleaned JSON output
    """
    result = result.split("}")
    output = json.loads("}".join(result[0:-1]) + "}")
    return output


def extract(messages: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
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
