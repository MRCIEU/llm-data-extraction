from local_funcs.funcs import calculate_start_end

data_length = 7000
pilot = False
array_length = 30
num_docs = 100

for task_id in range(array_length):
    startpoint, endpoint = calculate_start_end(
        array_task_id=task_id,
        array_length=array_length,
        num_docs=num_docs,
        data_length=data_length,
        pilot=pilot,
    )
    print(f"Task ID {task_id}: startpoint={startpoint}, endpoint={endpoint}")
