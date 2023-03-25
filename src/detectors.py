import cv2
import numpy as np
from abc import ABC, abstractmethod
from .utils import logger

class BaseDetector(ABC):
    """Abstract base for detectors."""
    
    def __init__(self, config):
        self.config = config
        self.hsv_lower = np.array(config.get('hsv', {}).get('lower_blue', [100, 50, 50]))
        self.hsv_upper = np.array(config.get('hsv', {}).get('upper_blue', [130, 255, 255]))
    
    @abstractmethod
    def detect(self, image):
        """Return contours or bboxes."""
        pass
    
    def get_vertices_2d(self, contour):
        """Approx 4 vertices from contour (shared logic)."""
        epsilon = self.config['contour']['epsilon'] * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / h
            if self.config['contour']['aspect_min'] <= aspect <= self.config['contour']['aspect_max']:
                return approx.reshape(4, 2).astype(np.int32)
        return np.array([])

class ColorCubeDetector(BaseDetector):
    """HSV-based blue cube detector (merges P13, P20, P45, etc.)."""
    
    def __init__(self, config, color='blue'):
        super().__init__(config)
        if color == 'red':
            self.hsv_lower = np.array(config.get('hsv', {}).get('lower_red', [0, 150, 50]))
            self.hsv_upper = np.array(config.get('hsv', {}).get('upper_red', [10, 255, 255]))
    
    def detect(self, image):
        """Return largest valid contour and mask."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Noise reduction (from P13)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.config['contour']['min_area']]
        
        if valid_contours:
            largest = max(valid_contours, key=cv2.contourArea)
            vertices = self.get_vertices_2d(largest)
            if len(vertices) > 0:
                return vertices, largest, mask
        return np.array([]), None, None

class YOLODetector(BaseDetector):
    """YOLOv8-based general object detector (upgraded from yolo_object_detection_main.py)."""
    
    def __init__(self, config):
        super().__init__(config)
        from ultralytics import YOLO
        self.model = YOLO(config['yolo']['model'])
    
    def detect(self, image):
        """Return bboxes and classes."""
        results = self.model(image, conf=self.config['yolo']['conf_threshold'], iou=self.config['yolo']['iou_threshold'])
        bboxes = []
        classes = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    if conf > self.config['yolo']['conf_threshold']:
                        bboxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])  # [x, y, w, h]
                        classes.append(cls)
        return bboxes, classes