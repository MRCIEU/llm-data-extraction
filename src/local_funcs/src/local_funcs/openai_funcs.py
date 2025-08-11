def get_o4_mini_result(client, input_prompt):
    response = client.responses.create(
        model="o4-mini",
        input=input_prompt,
        reasoning={"effort": "medium"},
    )
    output_text = response.output_text
    return output_text


def get_gpt_4o_result(client, input_prompt):
    response = client.responses.create(
        model="gpt-4o",
        input=input_prompt,
    )
    output_text = response.output_text
    return output_text


def get_gpt_4_1_result(client, input_prompt):
    response = client.responses.create(
        model="gpt-4.1",
        input=input_prompt,
    )
    output_text = response.output_text
    return output_text


def get_gpt_5_result(client, input_prompt):
    response = client.responses.create(
        model="gpt-5",
        input=input_prompt,
    )
    output_text = response.output_text
    return output_text


def get_gpt_5_mini_result(client, input_prompt):
    response = client.responses.create(
        model="gpt-5-mini",
        input=input_prompt,
    )
    output_text = response.output_text
    return output_text
