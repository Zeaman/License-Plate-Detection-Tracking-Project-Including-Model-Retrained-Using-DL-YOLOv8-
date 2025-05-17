#!/usr/bin/env python3
import os
import cv2
import glob
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sort import *
from utils import read_license_no


class LicensePlateDetector:
    def __init__(self):
        self.vehicle_model = YOLO("yolov8n.pt")
        self.plate_model = YOLO('./models/best.pt')
        self.tracker = Sort()
        self.cap = cv2.VideoCapture('./sample_video/License_Plate_Detection_Test.mp4')
        self.results = {}

        self.vehicle_plate_texts = {}
        self.vehicle_color = (255, 0, 0)  # Blue for vehicles without plates
        self.locked_vehicle_color = (0, 255, 255)  # Yellow for vehicles with plates
        self.plate_color = (0, 255, 0)  # Green for plates

        self.white = (255, 255, 255)
        self.red = (0, 0, 255)
        self.black = (0, 0, 0)

        self.box_thickness = 2
        self.text_thickness = 1
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7

        self.vehicle_classes = [2, 3, 5, 7]  # Car, motorcycle, bus, truck

        self.min_confidence = 0.4
        self.confidence_threshold = 0.1
        self.min_update_frames = 5

    def detect_and_display(self):
        frame_count = -1
        window_width = 1200
        window_height = 920

        while True:
            frame_count += 1
            ret, frame = self.cap.read()
            if not ret:
                break

            self.results[frame_count] = {}

            # Detect all vehicles
            vehicles = self.detect_vehicles(frame)
            tracked_vehicles = self.tracker.update(np.asarray(vehicles))
            visible_ids = set()

            # Detect license plates
            self.detect_license_plates(frame, tracked_vehicles, frame_count, visible_ids)
            
            # Draw all elements
            self.draw_labels(frame, tracked_vehicles, visible_ids, frame_count)

            cv2.namedWindow('License Plate Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('License Plate Detection', window_width, window_height)
            cv2.imshow('License Plate Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.save_results()

    def detect_vehicles(self, frame):
        detections = self.vehicle_model(frame)[0]
        vehicles = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in self.vehicle_classes and score > 0.4:  # Added confidence threshold
                vehicles.append([x1, y1, x2, y2, score])
        return vehicles

    def detect_license_plates(self, frame, tracked_vehicles, frame_count, visible_ids):
        plates = self.plate_model(frame)[0]
        for plate in plates.boxes.data.tolist():
            x1, y1, x2, y2, score, _ = plate
            if score < 0.5:  # Skip low-confidence detections
                continue
                
            vehicle = self.get_vehicle(plate, tracked_vehicles)
            vid = int(vehicle[-1])
            
            if vid == -1:
                continue
                
            visible_ids.add(vid)
            plate_img = frame[int(y1):int(y2), int(x1):int(x2)]
            plate_text, confidence = self.read_plate(plate_img)
            
            if plate_text and confidence > self.min_confidence:
                self.store_results(frame_count, vehicle, plate, plate_text, confidence)

    def draw_labels(self, frame, tracked_vehicles, visible_ids, frame_count):
        # First draw all vehicle bounding boxes
        for vehicle in tracked_vehicles:
            vx1, vy1, vx2, vy2, vid = map(int, vehicle)
            
            # Check if this vehicle has a locked plate
            if vid in self.vehicle_plate_texts:
                # Use yellow for vehicles with plates
                color = self.locked_vehicle_color
                frames_since_update = frame_count - self.vehicle_plate_texts[vid].get('last_updated', 0)
                if frames_since_update > self.min_update_frames:
                    # Draw thicker box for confirmed plates
                    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), color, self.box_thickness + 1)
                else:
                    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), color, self.box_thickness)
            else:
                # Use blue for vehicles without plates
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), self.vehicle_color, self.box_thickness)

        # Then draw plate information
        current_vehicle_plate_texts = {
            vid: self.vehicle_plate_texts[vid]
            for vid in visible_ids if vid in self.vehicle_plate_texts
        }
        self.vehicle_plate_texts = current_vehicle_plate_texts

        for vid, info in self.vehicle_plate_texts.items():
            plate_text = info['text']
            bbox = info['plate_bbox']
            if not bbox:
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw plate rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.plate_color, self.box_thickness)

            # Prepare license plate label
            ln_label = "LN:"
            plate_label = plate_text
            ln_size, _ = cv2.getTextSize(ln_label, self.font, self.font_scale, 2)
            plate_size, _ = cv2.getTextSize(plate_label, self.font, self.font_scale, 2)

            padding = 5
            spacing = 2
            total_width = ln_size[0] + plate_size[0] + 3 * padding + spacing
            height = max(ln_size[1], plate_size[1]) + 2 * padding

            box_top = y1 - height - 5
            box_bottom = y1 - 5
            box_right = x2
            box_left = x2 - total_width
            ln_box_right = box_left + ln_size[0] + 2 * padding

            # Draw label background and text
            cv2.rectangle(frame, (box_left, box_top), (ln_box_right, box_bottom), self.red, -1)
            cv2.putText(frame, ln_label, (box_left + padding, box_bottom - padding),
                        self.font, self.font_scale, self.white, 2, cv2.LINE_AA)
            cv2.rectangle(frame, (ln_box_right + spacing, box_top), (box_right, box_bottom), self.white, -1)
            cv2.putText(frame, plate_label, (ln_box_right + spacing + padding, box_bottom - padding),
                        self.font, self.font_scale, self.black, 2, cv2.LINE_AA)

    def get_vehicle(self, plate, vehicles):
        x1, y1, x2, y2, _, _ = plate
        for vehicle in vehicles:
            vx1, vy1, vx2, vy2, vid = vehicle
            if (x1 > vx1 and y1 > vy1 and x2 < vx2 and y2 < vy2):
                return vehicle
        return (-1, -1, -1, -1, -1)

    def read_plate(self, plate_img):
        if len(plate_img.shape) == 3:
            plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(plate_img, 68, 255, cv2.THRESH_BINARY_INV)
        return read_license_no(thresh)

    def store_results(self, frame_num, vehicle, plate, text, confidence):
        vx1, vy1, vx2, vy2, vid = vehicle
        px1, py1, px2, py2, pscore, _ = plate

        self.results[frame_num][vid] = {
            'vehicle': {'bbox': [vx1, vy1, vx2, vy2]},
            'plate': {
                'bbox': [px1, py1, px2, py2],
                'text': text,
                'score': confidence
            }
        }

        if vid not in self.vehicle_plate_texts:
            if confidence >= self.min_confidence:
                self.vehicle_plate_texts[vid] = {
                    'text': text,
                    'plate_bbox': [px1, py1, px2, py2],
                    'score': confidence,
                    'first_frame': frame_num,
                    'last_updated': frame_num,
                    'update_count': 1
                }
        else:
            current_info = self.vehicle_plate_texts[vid]
            frames_since_update = frame_num - current_info.get('last_updated', 0)
            if ((confidence - current_info['score']) >= self.confidence_threshold and
                    frames_since_update >= self.min_update_frames):
                self.vehicle_plate_texts[vid].update({
                    'text': text,
                    'plate_bbox': [px1, py1, px2, py2],
                    'score': confidence,
                    'last_updated': frame_num,
                    'update_count': current_info.get('update_count', 0) + 1
                })

            self.vehicle_plate_texts[vid]['plate_bbox'] = [px1, py1, px2, py2]

    def save_results(self):
        os.makedirs('./results', exist_ok=True)
        
        # Find the next available CSV filename
        csv_index = 1
        while True:
            csv_path = f'./results/recognized_plates_{csv_index}.csv'
            if not os.path.exists(csv_path):
                break
            csv_index += 1

        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('frame,vehicle_id,vehicle_bbox,plate_bbox,plate_text,confidence\n')
            for frame_num, data in self.results.items():
                for vid, detections in data.items():
                    vehicle = detections['vehicle']
                    plate = detections['plate']
                    f.write(
                        f"{frame_num},{vid},"
                        f"\"[{vehicle['bbox'][0]} {vehicle['bbox'][1]} {vehicle['bbox'][2]} {vehicle['bbox'][3]}]\","
                        f"\"[{plate['bbox'][0]} {plate['bbox'][1]} {plate['bbox'][2]} {plate['bbox'][3]}]\","
                        f"\"{plate['text']}\",{plate['score']}\n"
                    )
        print(f"Results saved to {csv_path}")
        return csv_path  # Return the path for the render_final_video function


def get_next_output_filename(base_dir="./video_output", prefix="output", ext="mp4"):
    os.makedirs(base_dir, exist_ok=True)
    i = 1
    while True:
        candidate = os.path.join(base_dir, f"{prefix}_{i}.{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def render_final_video(csv_path=None, video_path='./sample_video/License_Plate_Detection_Test.mp4'):
    # If no CSV path provided, find the most recent one
    if csv_path is None:
        csv_files = sorted(glob.glob('./results/recognized_plates_*.csv'))
        if not csv_files:
            print("No CSV files found in ./results directory")
            return
        csv_path = csv_files[-1]  # Use the most recent one
    
    print(f"Using CSV file: {csv_path}")
    output_path = get_next_output_filename()

    def parse_bbox(bbox_str):
        try:
            bbox_str = bbox_str.strip().strip('"[]')
            values = list(map(float, bbox_str.split()))
            if len(values) == 4:
                return values
        except Exception as e:
            print(f"Error parsing bbox: {bbox_str}, {e}")
        return [0.0, 0.0, 0.0, 0.0]

    try:
        results = pd.read_csv(csv_path)
        if 'frame' not in results.columns:
            print("Error: CSV file is missing 'frame' column")
            return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Group by frame and vehicle_id to get the best detection per vehicle per frame
    vehicle_data = results.groupby(['frame', 'vehicle_id']).first().reset_index()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Colors and styles
    vehicle_color = (255, 0, 0)  # Blue for vehicles without plates
    locked_vehicle_color = (0, 255, 255)  # Yellow for vehicles with plates
    plate_color = (0, 255, 0)  # Green for plates
    red = (0, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)
    box_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    text_thickness = 2
    padding = 5
    spacing = 2

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_vehicles = vehicle_data[vehicle_data['frame'] == frame_index]
        
        # First draw all vehicles
        for _, row in frame_vehicles.iterrows():
            vehicle_bbox = parse_bbox(row['vehicle_bbox'])
            vx1, vy1, vx2, vy2 = map(int, vehicle_bbox)
            
            # Check if this vehicle has a plate
            if not pd.isna(row['plate_text']):
                # Vehicle with plate - use yellow
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), locked_vehicle_color, box_thickness)
            else:
                # Vehicle without plate - use blue
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), vehicle_color, box_thickness)
        
        # Then draw plates and labels for vehicles with plates
        for _, row in frame_vehicles.iterrows():
            if pd.isna(row['plate_text']):
                continue
                
            plate_bbox = parse_bbox(row['plate_bbox'])
            plate_text = str(row['plate_text'])
            confidence = float(row['confidence'])
            
            px1, py1, px2, py2 = map(int, plate_bbox)
            cv2.rectangle(frame, (px1, py1), (px2, py2), plate_color, box_thickness)

            if plate_text != 'nan':
                ln_label = "LN:"
                label_plate = plate_text

                size_ln, _ = cv2.getTextSize(ln_label, font, font_scale, text_thickness)
                size_plate, _ = cv2.getTextSize(label_plate, font, font_scale, text_thickness)

                total_width = size_ln[0] + size_plate[0] + 3 * padding + spacing
                height = max(size_ln[1], size_plate[1]) + 2 * padding

                box_top = py1 - height - 5
                box_bottom = py1 - 5
                box_right = px2
                box_left = px2 - total_width
                ln_box_right = box_left + size_ln[0] + 2 * padding

                cv2.rectangle(frame, (box_left, box_top), (ln_box_right, box_bottom), red, -1)
                cv2.putText(frame, ln_label, (box_left + padding, box_bottom - padding),
                            font, font_scale, white, text_thickness, cv2.LINE_AA)
                cv2.rectangle(frame, (ln_box_right + spacing, box_top), (box_right, box_bottom), white, -1)
                cv2.putText(frame, label_plate, (ln_box_right + spacing + padding, box_bottom - padding),
                            font, font_scale, black, text_thickness, cv2.LINE_AA)

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    print(f"Annotated video saved to {output_path}")


if __name__ == "__main__":
    detector = LicensePlateDetector()
    csv_path = detector.detect_and_display()  # Save results and get CSV path
    render_final_video(csv_path=csv_path)  # Pass the CSV path directly