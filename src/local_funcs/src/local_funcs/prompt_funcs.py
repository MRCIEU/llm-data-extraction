from . import prompt_templates


def make_message_metadata(abstract: str):
    res = [
        {
            "role": "system",
            "content": "You are a data scientist responsible for extracting accurate information from research papers. "
            + "You answer each question with a single JSON string.",
        },
        {
            "role": "user",
            "content": f"""
                This is an abstract from a Mendelian randomization study.
                    "{abstract}"
                    """,
        },
        prompt_templates.metadataexample,
        prompt_templates.metadataprompt,
    ]
    return res


def make_message_results(abstract: str):
    res = [
        {
            "role": "system",
            "content": "You are a data scientist responsible for extracting accurate information from research papers. "
            + "You answer each question with a single JSON string.",
        },
        {
            "role": "user",
            "content": f"""
                This is an abstract from a Mendelian randomization study.
                    "{abstract}"
                    """,
        },
        prompt_templates.resultsexample,
        prompt_templates.resultsprompt,
    ]
    return res
