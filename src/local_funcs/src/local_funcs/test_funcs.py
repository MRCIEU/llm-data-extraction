from local_funcs.funcs import calculate_start_end


def test_pilot_mode_shorter_than_data():
    start, end = calculate_start_end(
        array_task_id=0,
        array_length=10,
        num_docs=5,
        data_length=20,
        pilot=True,
    )
    assert start == 0
    assert end == 5


def test_pilot_mode_truncated():
    start, end = calculate_start_end(
        array_task_id=0,
        array_length=10,
        num_docs=50,
        data_length=20,
        pilot=True,
    )
    assert start == 0
    assert end == 20


def test_normal_mode_within_bounds():
    start, end = calculate_start_end(
        array_task_id=1,
        array_length=2,
        num_docs=10,
        data_length=50,
        pilot=False,
    )
    # With array_length=2, data_length=50, chunk_size = 25
    # array_task_id=1: start=25, end=50
    assert start == 25
    assert end == 50


def test_normal_mode_endpoint_truncated():
    start, end = calculate_start_end(
        array_task_id=2,
        array_length=2,
        num_docs=10,
        data_length=25,
        pilot=False,
    )
    # With array_length=2, data_length=25, chunk_size=13
    # array_task_id=2: start=26, which is >= 25, so should return (None, None)
    assert start is None
    assert end is None


def test_normal_mode_endpoint_exact():
    start, end = calculate_start_end(
        array_task_id=1,
        array_length=2,
        num_docs=10,
        data_length=25,
        pilot=False,
    )
    # With array_length=2, data_length=25, chunk_size=13
    # array_task_id=1: start=13, end=25
    assert start == 13
    assert end == 25
