from local_funcs.funcs import calculate_chunk_start_end

data_length = 7000
pilot = False
array_length = 30
pilot_num_docs = 100

for task_id in range(array_length):
    startpoint, endpoint = calculate_chunk_start_end(
        chunk_id=task_id,
        num_chunks=array_length,
        data_length=data_length,
        pilot_num_docs=pilot_num_docs,
        pilot=pilot,
    )
    print(f"Task ID {task_id}: startpoint={startpoint}, endpoint={endpoint}")
