import cv2
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path='yolov8s.pt'):
        """
        YOLO Safe Initialization.
        - model.to() is NOT called manually
        - Ultralytics handles device automatically
        - yolov8s.pt will be auto-downloaded if missing
        """
        self.model = YOLO(model_path)

    def detect_and_crop(self, image):
        """Run inference with strict NMS, noise filtering, and duplicate removal."""
        # 1. APPLY NON-MAX SUPPRESSION (NMS) PROPERLY
        # Set: conf=0.6, iou=0.45, max_det=10 as required
        results = self.model.predict(
            source=image,
            conf=0.6,
            iou=0.45,
            max_det=10,
            verbose=False
        )
        
        raw_boxes = []
        img_h, img_w, _ = image.shape

        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                cls = int(box.cls[0].item())
                x1, y1, x2, y2 = [int(c) for c in coords]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                raw_boxes.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': conf,
                    'class': cls
                })

        # Debug Visual: Print raw count
        print(f"Before filtering: {len(raw_boxes)}")

        # 2. REMOVE SMALL / NOISY BOXES & OVERLAPPING DUPLICATES
        # Sort by confidence high to low
        sorted_boxes = sorted(raw_boxes, key=lambda x: x['conf'], reverse=True)
        final_detections = []
        
        MIN_AREA = 1000 # threshold for noise
        
        for cur in sorted_boxes:
            x1, y1, x2, y2 = cur['bbox']
            area = (x2 - x1) * (y2 - y1)
            
            # Filter by area
            if area < MIN_AREA:
                continue
            
            # 3. REMOVE OVERLAPPING DUPLICATE BOXES (IoU > 0.7)
            is_duplicate = False
            for kept in final_detections:
                iou = self._calculate_iou(cur['bbox'], kept['bbox'])
                if iou > 0.7:
                    # Optional Class-aware filter
                    if cur['class'] == kept['class']:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                crop = image[y1:y2, x1:x2]
                final_detections.append({
                    'bbox': [x1, y1, x2, y2], 
                    'crop': crop,
                    'det_conf': cur['conf'],
                    'class': cur['class']
                })

        # Debug Visual: Print after count
        print(f"After filtering: {len(final_detections)}")
        return final_detections

    def _calculate_iou(self, box1, box2):
        """Standard Intersection over Union calculation."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def draw_bbox(self, image, bbox, label="Object", color=(0, 255, 0)):
        x1, y1, x2, y2 = bbox
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Add label with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Position label: prefer top of box, else inside if at the very top of image
        label_y = y1 - 10 if y1 - 10 > h else y1 + h + 10
        
        # Background rectangle for text
        cv2.rectangle(image, (x1, label_y - h - 5), (x1 + w + 10, label_y + 5), color, -1)
        
        # White text for readability
        cv2.putText(image, label, (x1 + 5, label_y), font, font_scale, (255, 255, 255), thickness)
        return image
