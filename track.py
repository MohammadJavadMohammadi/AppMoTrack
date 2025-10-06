import os
import cv2
import joblib
import numpy as np
from tqdm import tqdm
from pathlib import Path
from tracker.appmotrack import AppMoTrack

class Args:
    def __init__(self):
        # Tracking thresholds
        self.track_high_thresh = 0.6
        self.track_low_thresh = 0.1
        self.new_track_thresh = 0.7
        self.track_buffer = 30
        
        # Association thresholds
        self.proximity_thresh = 0.5
        self.appearance_thresh = 0.25
        self.match_thresh = 0.8
        
        # Other parameters
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10
        self.mot20 = False # Set to True for MOT20 dataset
        
        # Pre-trained models
        self.ipca_path = "tracker/ipca_model.joblib"
        self.lda_path = "tracker/lda.pkl"
        
        # Load models
        print("Loading IPCA and LDA models...")
        self.loaded_ipca = joblib.load(self.ipca_path)
        self.lda_model = joblib.load(self.lda_path)
        print("Models loaded successfully.")

def main():
    # --- Configuration ---
    args = Args()
    frame_rate = 30
    
    # Paths
    dataset_base_dir = Path("datasets/MOT17/train")
    detections_base_dir = Path("detections/YOLOX-X/train")
    output_base_dir = Path("output")
    
    sequence_name = "MOT17-04-FRCNN" # Example sequence
    
    image_dir = dataset_base_dir / sequence_name / "img1"
    detection_file = detections_base_dir / f"{sequence_name}.txt"
    output_dir = output_base_dir / "mot_results"
    output_file = output_dir / f"{sequence_name}.txt"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Load Detections ---
    print(f"Loading detections from: {detection_file}")
    detections = {}
    with open(detection_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            x, y, w, h, conf = map(float, parts[1:6])
            if frame_id not in detections:
                detections[frame_id] = []
            detections[frame_id].append([x, y, x + w, y + h, conf, 0]) # x1, y1, x2, y2, conf, cls
    
    # --- Initialize Tracker ---
    tracker = AppMoTrack(args, frame_rate=frame_rate)
    
    # --- Run Tracking ---
    image_files = sorted(image_dir.glob("*.jpg"))
    results = []
    
    print(f"Starting tracking on sequence: {sequence_name}")
    for frame_id, img_path in enumerate(tqdm(image_files, desc="Processing frames"), 1):
        frame = cv2.imread(str(img_path))
        
        # Get detections for the current frame
        dets = np.array(detections.get(frame_id, []))
        if dets.size == 0:
            dets = np.empty((0, 6))
            
        # Update tracker
        online_targets = tracker.update(dets, frame)
        
        # Format results
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            results.append(
                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
            )

    # --- Save Results ---
    with open(output_file, 'w') as f:
        f.writelines(results)
        
    print(f"Tracking finished. Results saved to {output_file}")

if __name__ == "__main__":
    main()
