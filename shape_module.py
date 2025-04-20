import numpy as np
import cv2

class ShapeGenerator:
    def __init__(self):
        self.canvas_size = (512, 512)
        self.shapes = {
            "사각형": self._draw_rectangle,
            "원": self._draw_circle,
            "삼각형": self._draw_triangle
        }
    
    def _draw_rectangle(self, size=0.7):
        canvas = np.ones((*self.canvas_size, 3), dtype=np.uint8) * 255
        center = (self.canvas_size[0] // 2, self.canvas_size[1] // 2)
        side = int(min(self.canvas_size) * size)
        half_side = side // 2
        
        pt1 = (center[0] - half_side, center[1] - half_side)
        pt2 = (center[0] + half_side, center[1] + half_side)
        
        # 흰색 캔버스에 검은색 윤곽선
        cv2.rectangle(canvas, pt1, pt2, (0, 0, 0), 2)
        return canvas
    
    def _draw_circle(self, size=0.7):
        canvas = np.ones((*self.canvas_size, 3), dtype=np.uint8) * 255
        center = (self.canvas_size[0] // 2, self.canvas_size[1] // 2)
        radius = int(min(self.canvas_size) * size // 2)
        
        cv2.circle(canvas, center, radius, (0, 0, 0), 2)
        return canvas
    
    def _draw_triangle(self, size=0.7):
        canvas = np.ones((*self.canvas_size, 3), dtype=np.uint8) * 255
        center = (self.canvas_size[0] // 2, self.canvas_size[1] // 2)
        side = int(min(self.canvas_size) * size)
        
        height = int(side * np.sqrt(3) / 2)
        half_side = side // 2
        
        pt1 = (center[0], center[1] - height//2)
        pt2 = (center[0] - half_side, center[1] + height//2)
        pt3 = (center[0] + half_side, center[1] + height//2)
        
        pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
        
        cv2.polylines(canvas, [pts], True, (0, 0, 0), 2)
        return canvas
    
    def generate_shape(self, shape_name):
        shape_name = shape_name.lower()
        
        # 한국어 형태명 매핑
        if "사각형" in shape_name or "네모" in shape_name:
            shape_func = self.shapes["사각형"]
        elif "원" in shape_name:
            shape_func = self.shapes["원"]
        elif "삼각형" in shape_name or "세모" in shape_name:
            shape_func = self.shapes["삼각형"]
        else:
            # 기본값은 사각형
            shape_func = self.shapes["사각형"]
            
        return shape_func()