import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from datasets import load_dataset
from get_repo_structure.get_repo_structure import get_project_structure_from_scratch


def process_bug(bug):
    instance_id = bug['instance_id']

    json_file_path = f"./repo_structures/{instance_id}.json"

    if os.path.exists(json_file_path):
        return f"File {json_file_path} already exists, skipping this instance."

    d = get_project_structure_from_scratch(
        bug["repo"], bug["base_commit"], instance_id, "playground"
    )

    with open(json_file_path, "w") as json_file:
        json.dump(d, json_file, indent=4, ensure_ascii=False)

    return f"Saved {json_file_path}"


swe_bench_data = load_dataset("NoCode-bench/NoCode-bench_Verified", split="test")

if __name__ == '__main__':
    num_workers = min(os.cpu_count(), 32)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_bug = {executor.submit(process_bug, bug): bug for bug in swe_bench_data}

        for future in tqdm(as_completed(future_to_bug), total=len(swe_bench_data)):
            try:
                future.result()
            except Exception as exc:
                instance_id = future_to_bug[future]['instance_id']
                print(f"Exception occurred while processing instance {instance_id}: {exc}")

    print("All instances processed.")
