Here’s a detailed description of how to install requirements and use the project, formatted for inclusion in your **README.md** file:

---

# CaféVision: Real-Time Table Occupancy Detection

## Overview
CaféVision is a real-time table occupancy detection system leveraging YOLOv11 to classify restaurant tables as "occupied" or "unoccupied" based on proximity analysis of patrons in video feeds. This cost-effective and scalable solution eliminates the need for physical sensors, enhancing operational efficiency and customer satisfaction.

---

## Features
- **Real-Time Monitoring**: Detects table occupancy in real-time.
- **High Accuracy**: Utilizes YOLOv11x for robust object detection.
- **Scalable and Cost-Effective**: Requires no additional hardware, leveraging existing surveillance systems.

---

## Installation

### Prerequisites
1. Install [Python 3.8+](https://www.python.org/downloads/).
2. Install pip (Python's package installer).

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/CafeVision.git
   cd CafeVision
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Prepare Input Videos**:
   - Place your test videos in the `datasets/videos/` directory. For example:
     - `datasets/videos/RestaurantTest.mp4`
     - `datasets/videos/RestaurantTestEmpty.mp4`

2. **Run the Detection Script**:
   ```bash
   python src/DetectionYOLOVv11.py
   ```

3. **Output Videos**:
   - Processed videos with bounding boxes indicating "occupied" and "unoccupied" tables will be saved in the `results/` directory. For example:
     - `results/output_video1.mp4`

4. **Customize Parameters**:
   - You can adjust parameters like confidence thresholds and IoU thresholds directly in the `DetectionYOLOVv11x.py` script for better results in specific environments.

---

## File Structure
```
CafeVision/
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── src/
│   ├── DetectionYOLOVv11.py  # Main detection script
├── datasets/
│   ├── videos/
│       ├── RestaurantTest.mp4
│       ├── RestaurantTestEmpty.mp4
├── results/
│   ├── output_video1.mp4
│   ├── output_video2.mp4
├── .gitignore              # Ignore unnecessary files
```

---

## Customization
- **Confidence Thresholds**:
  - Modify `CONFIDENCE_THRESHOLD` in `DetectionYOLOVv11.py` to tune the model's sensitivity.
- **IoU Threshold**:
  - Adjust `IOU_THRESHOLD` for better overlapping bounding box handling.
- **Proximity Threshold**:
  - Customize `DISTANCE_THRESHOLD_PIXELS` to adapt to your specific restaurant layout.

---

## License
This project is licensed under the MIT License.

---
