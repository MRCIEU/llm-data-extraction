import json

from yiutils.project_utils import find_project_root


def load_schema_data():
    PROJECT_ROOT = find_project_root("justfile")
    DATA_DIR = PROJECT_ROOT / "data"
    PATH_SCHEMA_DIR = DATA_DIR / "assets" / "data-schema" / "example-data"
    assert PATH_SCHEMA_DIR.exists(), f"Schema directory not found: {PATH_SCHEMA_DIR}"

    schema_config = {
        "metadata": {
            "example": PATH_SCHEMA_DIR / "metadata.json",
            "schema": PATH_SCHEMA_DIR / "metadata.schema.json",
        },
        "results": {
            "example": PATH_SCHEMA_DIR / "results.json",
            "schema": PATH_SCHEMA_DIR / "results.schema.json",
        },
    }
    # Check if the four schema files exist
    missing_files = []
    schema_data = {}
    for section_name, section in schema_config.items():
        schema_data[section_name] = {}
        for key, path in section.items():
            if not path.exists():
                missing_files.append(str(path))
                schema_data[section_name][key] = None
            else:
                with path.open("r") as f:
                    try:
                        schema_data[section_name][key] = json.load(f)
                    except Exception as e:
                        print(f"ERROR loading {path}: {e}")
                        schema_data[section_name][key] = None
    if missing_files:
        print(f"WARNING: The following schema files do not exist: {missing_files}")
    else:
        print("All schema files found.")
    return schema_data
