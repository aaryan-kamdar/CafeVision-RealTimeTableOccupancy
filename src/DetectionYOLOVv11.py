import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Constants 
CONFIDENCE_THRESHOLD = 0.20 # for best results 0.2for restuaranttest.mp4 and 0.35 for restuaranttestempty.mp4
DISTANCE_THRESHOLD_PIXELS = 120  # Proximity threshold for person-to-table
IOU_THRESHOLD = 0.45  # IoU threshold for Non-Maximum Suppression (NMS)
ASSOCIATION_TOLERANCE_PIXELS = 50  # Tolerance for table shifts due to camera movement
DETECTION_INTERVAL = 5  # Frames between re-detection
FRAME_OCCUPIED_THRESHOLD = 3  # Frames to mark as occupied
FRAME_UNOCCUPIED_THRESHOLD = 8  # Frames to mark as unoccupied
MISSING_FRAME_THRESHOLD = 4  # Frames before removing undetected table
DEFAULT_COLOR = (0, 255, 0)  # Green for unoccupied tables
OCCUPIED_COLOR = (0, 0, 255)  # Red for occupied tables

# Load YOLO model
model = YOLO("yolo11x.pt")

# Input video path
video_path = "RestaurantTest.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize Video Writer
output_path = "output_video3.mp4"
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Table tracking data
table_data = defaultdict(lambda: {
    "bounding_box": None,
    "occupied": False,
    "occupied_frame_count": 0,
    "unoccupied_frame_count": 0,
    "missing_frame_count": 0,
})

frame_count = 0

def iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes."""
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2

    # Intersection
    xi1, yi1 = max(x1, x1p), max(y1, y1p)
    xi2, yi2 = min(x2, x2p), min(y2, y2p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Union
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2p - x1p) * (y2p - y1p)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def non_max_suppression(boxes, iou_threshold):
    """Apply Non-Maximum Suppression (NMS) to remove overlapping bounding boxes."""
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)  # Sort by confidence
    selected_boxes = []

    while boxes:
        current = boxes.pop(0)
        selected_boxes.append(current)
        boxes = [box for box in boxes if iou(current[:4], box[:4]) < iou_threshold]

    return selected_boxes

def associate_tables(existing_tables, detected_tables):
    """Associate tracked tables with newly detected ones."""
    associations = {}
    unmatched_detected = set(range(len(detected_tables)))

    for i, existing in enumerate(existing_tables):
        if existing["bounding_box"] is None:
            continue

        best_match = -1
        best_distance = ASSOCIATION_TOLERANCE_PIXELS
        for j, detected in enumerate(detected_tables):
            x1, y1, x2, y2 = existing["bounding_box"]
            x1d, y1d, x2d, y2d = detected
            center_existing = ((x1 + x2) // 2, (y1 + y2) // 2)
            center_detected = ((x1d + x2d) // 2, (y1d + y2d) // 2)
            distance = np.sqrt((center_existing[0] - center_detected[0]) ** 2 + (center_existing[1] - center_detected[1]) ** 2)

            if distance < best_distance:
                best_match = j
                best_distance = distance

        if best_match != -1:
            associations[i] = detected_tables[best_match]
            unmatched_detected.discard(best_match)

    return associations, unmatched_detected

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Re-detect objects every DETECTION_INTERVAL frames
    if frame_count % DETECTION_INTERVAL == 0:
        results = model(frame, conf=CONFIDENCE_THRESHOLD)

        # Reset detected tables
        detected_tables = []

        for result in results[0].boxes:
            cls = int(result.cls)
            conf = result.conf
            x1, y1, x2, y2 = map(int, result.xyxy[0])

            if conf > CONFIDENCE_THRESHOLD and model.names[cls] == "dining table":
                detected_tables.append((x1, y1, x2, y2, conf))

        # Apply Non-Maximum Suppression
        detected_tables = non_max_suppression(detected_tables, IOU_THRESHOLD)

        # Remove confidence scores
        detected_tables = [box[:4] for box in detected_tables]

        # Associate new detections with existing tables
        existing_tables = list(table_data.values())
        associations, unmatched = associate_tables(existing_tables, detected_tables)

        # Update existing table data
        for i, detected_bbox in associations.items():
            table_data[i]["bounding_box"] = detected_bbox

        # Mark tables as missing if not matched with new detections
        for i in range(len(existing_tables)):
            if i not in associations:
                table_data[i]["missing_frame_count"] += 1
            else:
                table_data[i]["missing_frame_count"] = 0

        # Add unmatched detections as new tables
        for unmatched_idx in unmatched:
            table_data[len(table_data)] = {
                "bounding_box": detected_tables[unmatched_idx],
                "occupied": False,
                "occupied_frame_count": 0,
                "unoccupied_frame_count": 0,
                "missing_frame_count": 0,
            }

        # Remove tables that have been missing for too long
        keys_to_remove = [key for key, table in table_data.items() if table["missing_frame_count"] > MISSING_FRAME_THRESHOLD]
        for key in keys_to_remove:
            del table_data[key]

    # Detect people and check proximity to tables
    results = model(frame, conf=CONFIDENCE_THRESHOLD)
    person_boxes = []

    for result in results[0].boxes:
        cls = int(result.cls)
        conf = result.conf
        x1, y1, x2, y2 = map(int, result.xyxy[0])

        if conf > CONFIDENCE_THRESHOLD and model.names[cls] == "person":
            person_boxes.append((x1, y1, x2, y2))

    for i, table in table_data.items():
        if table["bounding_box"] is None:
            continue

        x1, y1, x2, y2 = table["bounding_box"]
        table_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Check proximity of persons to table
        near_person = any(
            np.sqrt((px - table_center[0]) ** 2 + (py - table_center[1]) ** 2) < DISTANCE_THRESHOLD_PIXELS
            for (px1, py1, px2, py2) in person_boxes
            for px, py in [((px1 + px2) // 2, (py1 + py2) // 2)]
        )

        # Update frame-based counters for occupancy logic
        if near_person:
            table["occupied_frame_count"] += 1
            table["unoccupied_frame_count"] = 0
            if table["occupied_frame_count"] >= FRAME_OCCUPIED_THRESHOLD:
                table["occupied"] = True
        else:
            table["unoccupied_frame_count"] += 1
            table["occupied_frame_count"] = 0
            if table["unoccupied_frame_count"] >= FRAME_UNOCCUPIED_THRESHOLD:
                table["occupied"] = False

    # Count occupied and unoccupied tables
    total_tables = len(table_data)
    occupied_count = sum(1 for table in table_data.values() if table["occupied"])
    unoccupied_count = total_tables - occupied_count

    # Visualize table statuses and counts
    for table in table_data.values():
        if table["bounding_box"] is None:
            continue

        x1, y1, x2, y2 = table["bounding_box"]
        color = OCCUPIED_COLOR if table["occupied"] else DEFAULT_COLOR
        status_text = "Occupied" if table["occupied"] else "Unoccupied"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display total, occupied, and unoccupied table counts
    count_text = f"Total: {total_tables} | Occupied: {occupied_count} | Unoccupied: {unoccupied_count}"
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow("Table Occupancy Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
