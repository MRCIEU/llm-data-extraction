from . import prompt_templates


def make_message_metadata(abstract: str):
    res = [
        prompt_templates.system_prompt,
        {
            "role": "user",
            "content": f"""
                This is an abstract from a Mendelian randomization study.
                    "{abstract}"
                    """,
        },
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
        prompt_templates.results_example,
        prompt_templates.results_prompt,
    ]
    return res
