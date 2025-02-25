import json
import os
import numpy as np
import argparse
import cv2
import pandas as pd
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_iou(boxA, boxB):
    """Compute Intersection over Union (IoU) for two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if (boxAArea + boxBArea - interArea) == 0:
        return 0
    return interArea / (boxAArea + boxBArea - interArea)

def compute_confusion_matrix(matched_tracks, gt_tracks, pipeline_tracks, output_folder):
    """Computes and saves confusion matrix."""
    gt_labels = []
    pred_labels = []

    for gt_id, pipe_id in matched_tracks.items():
        gt_labels.append(gt_tracks[gt_id]["label"])
        pred_labels.append(pipeline_tracks[pipe_id]["label"])

    unique_labels = list(set(gt_labels + pred_labels))
    cm = confusion_matrix(gt_labels, pred_labels, labels=unique_labels)

    # Save as JSON
    cm_json = {unique_labels[i]: {unique_labels[j]: int(cm[i, j]) for j in range(len(unique_labels))}
               for i in range(len(unique_labels))}
    with open(os.path.join(output_folder, "confusion_matrix.json"), "w") as f:
        json.dump(cm_json, f, indent=4)

    # Save as CSV
    df_cm = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    df_cm.to_csv(os.path.join(output_folder, "confusion_matrix.csv"))

    # Generate Heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(df_cm, annot=False, fmt="d", cmap="Blues", xticklabels=unique_labels,
                     yticklabels=unique_labels, linewidths=0.5, linecolor="gray",
                     cbar=False)

    # Highlight diagonal separately
    for i in range(len(unique_labels)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=3))

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))
    plt.close()

def extract_tracks(data):
    """Extracts tracks from annotation JSON, indexed by track ID."""
    tracks = {}
    
    # print(data)

    for track in data[0]["tracks"]:
        track_id = track["frame"]  # Unique track identifier
        label = track["label"]
        frames = {}
        
        for shape in track["shapes"]:
            frame = shape["frame"]
            bbox = shape["points"]
            frames[frame] = bbox  # Store bbox for each frame
        
        tracks[track_id] = {"label": label, "frames": frames}
    
    return tracks

def find_common_frame_range(gt_frames, pipeline_frames):
    """Finds the second and second-to-last common frames of a matched track."""
    common_frames = set(gt_frames).intersection(set(pipeline_frames))
    if not common_frames:
        return None, None
    if len(common_frames) == 1:
        return common_frames.pop(), common_frames.pop()
    return sorted(common_frames)[1], sorted(common_frames)[-2]

def match_tracks(gt_tracks, pipeline_tracks, iou_threshold=0.5):
    """Matches GT tracks with pipeline tracks using IoU."""
    matched_tracks = {}
    
    for gt_id, gt_data in gt_tracks.items():
        best_match = None
        best_iou = 0
        for pipe_id, pipe_data in pipeline_tracks.items():
            common_frames = set(gt_data["frames"]).intersection(set(pipe_data["frames"]))
            if not common_frames:
                continue
            
            ious = []
            for frame in common_frames:
                iou = calculate_iou(gt_data["frames"][frame], pipe_data["frames"][frame])
                ious.append(iou)
            
            avg_iou = np.mean(ious) if ious else 0
            if avg_iou > best_iou and avg_iou >= iou_threshold:
                best_iou = avg_iou
                best_match = pipe_id
        
        if best_match is not None:
            matched_tracks[gt_id] = best_match
    
    return matched_tracks

def compute_metrics(matched_tracks, gt_tracks, pipeline_tracks):
    """Computes detection and classification metrics."""
    per_class_metrics = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "Correct": 0, "Total": 0})

    for gt_id, gt_data in gt_tracks.items():
        gt_label = gt_data["label"]
        if gt_id in matched_tracks:
            pipe_id = matched_tracks[gt_id]
            pipeline_label = pipeline_tracks[pipe_id]["label"]
            per_class_metrics[gt_label]["TP"] += 1
            per_class_metrics[gt_label]["Total"] += 1
            if gt_label == pipeline_label:
                per_class_metrics[gt_label]["Correct"] += 1
        else:
            per_class_metrics[gt_label]["FN"] += 1
            per_class_metrics[gt_label]["Total"] += 1

    for pipe_id, pipe_data in pipeline_tracks.items():
        if pipe_id not in matched_tracks.values():
            per_class_metrics[pipe_data["label"]]["FP"] += 1

    # Compute final per-class precision, recall, F1-score, and classification accuracy
    per_class_results = {}
    for label, stats in per_class_metrics.items():
        TP, FP, FN, Correct, Total = stats["TP"], stats["FP"], stats["FN"], stats["Correct"], stats["Total"]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = Correct / Total if Total > 0 else 0

        per_class_results[label] = {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score,
            "Classification Accuracy": accuracy,
            "TP": TP,
            "FP": FP,
            "FN": FN
        }

    return per_class_results

def save_metrics(per_class_metrics, output_folder):
    """Saves the computed metrics to CSV and JSON files."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Save per-class metrics as JSON
    with open(os.path.join(output_folder, "per_class_metrics.json"), "w") as f:
        json.dump(per_class_metrics, f, indent=4)

    # Remove per-class metrics for the "D3-1" class (only for debugging purposes)
    per_class_metrics.pop("D3-1", None)
    
    # Global metrics (weighted averages for precision, recall, F1-score, and classification accuracy and sums for TP, FP, FN)
    global_metrics = {
        "Precision": np.mean([stats["Precision"] for stats in per_class_metrics.values()]),
        "Recall": np.mean([stats["Recall"] for stats in per_class_metrics.values()]),
        "F1 Score": np.mean([stats["F1 Score"] for stats in per_class_metrics.values()]),
        "Classification Accuracy": np.mean([stats["Classification Accuracy"] for stats in per_class_metrics.values()]),
        "TP": sum(stats["TP"] for stats in per_class_metrics.values()),
        "FP": sum(stats["FP"] for stats in per_class_metrics.values()),
        "FN": sum(stats["FN"] for stats in per_class_metrics.values())
    }

    # Save global metrics as JSON
    with open(os.path.join(output_folder, "global_metrics.json"), "w") as f:
        json.dump(global_metrics, f, indent=4)

def save_debug_frames(video_path, output_folder, matched_tracks, gt_tracks, pipeline_tracks):
    """Extracts and saves the first and last frames of the common part of matched tracks."""
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    for gt_id, pipe_id in matched_tracks.items():
        gt_frames = gt_tracks[gt_id]["frames"]
        pipe_frames = pipeline_tracks[pipe_id]["frames"]
        
        start_frame, end_frame = find_common_frame_range(gt_frames.keys(), pipe_frames.keys())
        if start_frame is None or end_frame is None:
            continue
        
        for f in [start_frame, end_frame]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, img = cap.read()
            if ret:
                frame_file = os.path.join(output_folder, f"track_{gt_id}_frame_{f}.jpg")

                # Draw GT bounding box (red)
                x1_gt, y1_gt, x2_gt, y2_gt = map(int, gt_frames[f])
                cv2.rectangle(img, (x1_gt, y1_gt), (x2_gt, y2_gt), (0, 0, 255), 2)
                cv2.putText(img, "GT", (x1_gt, y1_gt - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Draw pipeline bounding box (green)
                x1_pipe, y1_pipe, x2_pipe, y2_pipe = map(int, pipe_frames[f])
                cv2.rectangle(img, (x1_pipe, y1_pipe), (x2_pipe, y2_pipe), (0, 255, 0), 2)
                cv2.putText(img, "Pipeline", (x1_pipe, y1_pipe - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imwrite(frame_file, img)

    cap.release()

def evaluate(pipeline_file, gt_file, output_folder, video_file=None, debug=False):
    """Main evaluation function."""
    os.makedirs(output_folder, exist_ok=True)
    
    with open(pipeline_file, "r") as f:
        pipeline_data = json.load(f)

    with open(gt_file, "r") as f:
        gt_data = json.load(f)

    gt_tracks = extract_tracks(gt_data)
    pipeline_tracks = extract_tracks(pipeline_data)

    matched_tracks = match_tracks(gt_tracks, pipeline_tracks, iou_threshold=0.5)
    per_class_metrics = compute_metrics(matched_tracks, gt_tracks, pipeline_tracks)

    save_metrics(per_class_metrics, output_folder)

    # Compute confusion matrix
    compute_confusion_matrix(matched_tracks, gt_tracks, pipeline_tracks, output_folder)

    if debug:    
        save_debug_frames(video_file, os.path.join(output_folder, "debug_frames"), matched_tracks, gt_tracks, pipeline_tracks)

def main():
    parser = argparse.ArgumentParser(description="Evaluate tracking results.")
    parser.add_argument("--video_file", type=str, required=False, help="Path to the video file.", default="./data/2024_05_20_22_21_56_519/task_backup_2024_05_20_22_21_56_519/data/2024_05_20_22_21_56_519_cam_2024_05_20_22_21_56_519.mp4")
    parser.add_argument("--pipeline_file", type=str, required=False, help="Path to pipeline annotations JSON.", default="./data/task_backup_2024_05_20_22_21_56_519/annotations_pipeline.json")
    parser.add_argument("--gt_file", type=str, required=False, help="Path to ground truth annotations JSON.", default="./data/2024_05_20_22_21_56_519 ANNOTATED/annotations_gt.json")
    parser.add_argument("--output_folder", type=str, required=False, help="Output directory for evaluation results.", default="./output")
    parser.add_argument("--debug", action="store_true", help="Save debug frames.", default=False)
    args = parser.parse_args()
    
    evaluate(args.pipeline_file, args.gt_file, args.video_file, args.output_folder, args.debug)

if __name__ == "__main__":
    main()