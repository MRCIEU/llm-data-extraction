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
    # start = 1*2*10 = 20, end = 20+10 = 30
    assert start == 20
    assert end == 30


def test_normal_mode_endpoint_truncated():
    start, end = calculate_start_end(
        array_task_id=2,
        array_length=2,
        num_docs=10,
        data_length=25,
        pilot=False,
    )
    # start = 2*2*10 = 40, which is > 25, so should return (None, None)
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
    # start = 20, end = 30, but data_length=25, so end should be 25
    assert start == 20
    assert end == 25
