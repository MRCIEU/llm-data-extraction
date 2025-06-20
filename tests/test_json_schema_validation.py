import json
import glob
import pytest
import jsonschema
from pathlib import Path

def get_schema_and_json_pairs():
    # Find all .json.schema files and their corresponding .json files
    base = Path(__file__).parent.parent.parent / "data" / "assets" / "data-schema"
    pairs = []
    for schema_path in base.rglob("*.json.schema"):
        json_path = schema_path.with_suffix("")  # Remove .schema
        if json_path.exists():
            pairs.append((schema_path, json_path))
    return pairs

@pytest.mark.parametrize("schema_path,json_path", get_schema_and_json_pairs())
def test_json_matches_schema(schema_path, json_path):
    with open(schema_path) as f:
        schema = json.load(f)
    with open(json_path) as f:
        data = json.load(f)
    # If the schema is for an array, validate each item
    if schema.get("type") == "array":
        assert isinstance(data, list), f"{json_path} should be a list"
        for item in data:
            jsonschema.validate(item, schema["items"])
    else:
        jsonschema.validate(data, schema)
