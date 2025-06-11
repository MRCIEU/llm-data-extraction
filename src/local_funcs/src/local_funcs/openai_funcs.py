from local_funcs import prompt_templates


def generate_message(abstract):
    messages = [
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
        prompt_templates.metadataexample,
        prompt_templates.metadataprompt,
    ]
    return messages


def get_o4_mini_result(client, abstract):
    input = generate_message(abstract)
    response = client.responses.create(
        model="o4-mini",
        input=input,
        reasoning={"effort": "medium"},
    )
    output_text = response.output_text
    return output_text


def get_gpt_4o_result(client, abstract):
    input = generate_message(abstract)
    response = client.responses.create(
        model="gpt-4o",
        input=input,
    )
    output_text = response.output_text
    return output_text
