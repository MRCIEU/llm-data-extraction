from . import prompt_templates

# ==== legacy design ====


def make_message_metadata(abstract: str):
    res = [
        prompt_templates.system_prompt,
        prompt_templates.make_abstract_input_prompt(abstract),
        prompt_templates.metadata_example,
        prompt_templates.metadata_prompt,
    ]
    return res


def make_message_results(abstract: str):
    res = [
        prompt_templates.system_prompt,
        {
            "role": "user",
            "content": f"""
                This is an abstract from a Mendelian randomization study.
                    "{abstract}"
                    """,
        },
        prompt_templates.make_abstract_input_prompt(abstract),
        prompt_templates.results_example,
        prompt_templates.results_prompt,
    ]
    return res


# ==== new design ====


def make_message_metadata_new(abstract: str, json_example: str, json_schema: str):
    res = [
        prompt_templates.system_prompt,
        prompt_templates.make_abstract_input_prompt(abstract),
        prompt_templates.make_example_output_prompt(json_example, json_schema),
        prompt_templates.metadata_prompt,
    ]
    return res


def make_message_results_new(abstract: str, json_example: str, json_schema: str):
    res = [
        prompt_templates.system_prompt,
        prompt_templates.make_abstract_input_prompt(abstract),
        prompt_templates.make_example_output_prompt(json_example, json_schema),
        prompt_templates.results_prompt,
    ]
    return res
