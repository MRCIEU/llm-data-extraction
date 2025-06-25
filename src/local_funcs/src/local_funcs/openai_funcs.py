# This is just a set of temp helper utils
from local_funcs import prompt_templates


def generate_message(abstract):
    messages = [
        prompt_templates.system_prompt,
        {
            "role": "user",
            "content": f"""
                This is an abstract from a Mendelian randomization study.
                    "{abstract["ab"]}"   """,
        },
        prompt_templates.metadata_example,
        prompt_templates.metadata_prompt,
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
