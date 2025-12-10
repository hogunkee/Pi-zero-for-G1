import os
import json
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser(description="LeRobot 데이터셋 병합")
parser.add_argument("--data_dir", type=str, default='/data1/hogun/dataset/1205_TaskDecompose')
args = parser.parse_args()

task_dict = {}
with open('tasks_decompose.jsonl', 'r', encoding='utf-8') as src_f:
    for line in src_f:
        x = json.loads(line)
        task_dict[x["task"]] = x["task_index"]

for task_name in os.listdir(args.data_dir):
    if task_name.replace("_", " ") in task_dict:
        task_index = task_dict[task_name.replace("_", " ")]
    else:
        continue
        
    episodes = sorted(os.listdir(os.path.join(data_dir, task_name, 'data/chunk-000')))
    for ep in episodes:
        parquet_file = os.path.join(data_dir, task_name, 'data/chunk-000', ep)
        data = pd.read_parquet(parquet_file)
        data['task_index'] = task_index
        data.to_parquet(parquet_file) # overwrite parquet files