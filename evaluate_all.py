import os
from main import evaluate
from tqdm import tqdm

root = "C:/Users/mouta/Desktop/GRA/CVAT/evaluate/data"

recordings_to_evaluates = os.listdir(root)

# Remove "zips" folder
recordings_to_evaluates.remove("zips")

for recording in tqdm(recordings_to_evaluates, desc="Evaluating recordings"):
    print(f"Evaluating recording '{recording}'...")
    
    subfolder = os.path.join(root, recording)

    # Find the pipeline folder (bgins with "task_")
    pipeline_folder = [f for f in os.listdir(subfolder) if f.startswith("task_")][0]
    pipeline_path = os.path.join(subfolder, pipeline_folder)
    
    # Find the pipeline file (ends with ".json")
    pipeline_file = [f for f in os.listdir(pipeline_path) if f.endswith(".json")][0]
    pipeline_file = os.path.join(pipeline_path, pipeline_file)

    # Find the GT folder (does not begin with "task_" or "output")
    gt_folder = [f for f in os.listdir(subfolder) if not f.startswith("task_") and not f.startswith("output")][0]
    gt_path = os.path.join(subfolder, gt_folder)

    # Find the GT file (ends with ".json")
    print(gt_path)
    gt_file = [f for f in os.listdir(gt_path) if f.endswith(".json")][0]
    gt_file = os.path.join(gt_path, gt_file)

    # Output folder
    output_folder = os.path.join(subfolder, "output")

    # Evaluate
    evaluate(pipeline_file, gt_file, output_folder)

print("Done!")