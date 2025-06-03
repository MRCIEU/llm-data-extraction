import json
import re


def extract_json_from_markdown(md_string):
    """
    Extracts and parses the first JSON object found in a markdown code block.
    Returns the parsed JSON object, or None if not found or invalid.

    Used for reasoning model output
    """
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, md_string, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None


def parse_json(json_string):
    """
    Parses a JSON string and returns the corresponding Python object.
    Returns None if the JSON is invalid.
    Used for reasoning model output
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return None


def extract_thinking(md_string):
    """
    Extracts the text before the first markdown JSON code block.
    Returns the extracted text, or an empty string if no JSON code block is found.

    Used for reasoning model output
    """
    import re

    pattern = r"(.*?)```json\s.*?```"
    match = re.search(pattern, md_string, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return md_string.strip()
