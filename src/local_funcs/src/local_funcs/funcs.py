def foobar():
    res = True
    return res


def calculate_start_end(
    array_task_id: int,
    array_length: int,
    num_docs: int,
    data_length: int,
    pilot: bool = False,
):
    """
    Calculate startpoint and endpoint for data processing.

    Parameters:
        array_task_id (int): The array index (e.g. from SLURM).
        array_length (int): The number of arrays (chunks) per job.
        num_docs (int): Number of documents per chunk.
        data_length (int): Total number of documents in the data.
        pilot (bool): If True, always return (0, num_docs) (capped by data_length).

    Returns:
        (startpoint, endpoint): Tuple of indices for slicing.
        If startpoint >= data_length, returns (None, None).
    """
    if pilot:
        startpoint = 0
        endpoint = min(num_docs, data_length)
        return startpoint, endpoint

    startpoint = array_task_id * array_length * num_docs
    endpoint = startpoint + num_docs
    if startpoint >= data_length:
        return None, None
    if endpoint > data_length:
        endpoint = data_length
    return startpoint, endpoint
