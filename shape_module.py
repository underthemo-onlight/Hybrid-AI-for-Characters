import numpy as np
import cv2

class ShapeGenerator:
    def __init__(self):
        self.canvas_size = (512, 512)
        self.shapes = {
            "사각형": self._draw_rectangle,
            "정사각형": self._draw_square,
            "원": self._draw_circle,
            "삼각형": self._draw_triangle,
            "별": self._draw_star
        }
    
    def _draw_rectangle(self, size=0.7):
        canvas = np.ones((*self.canvas_size, 3), dtype=np.uint8) * 255
        center = (self.canvas_size[0] // 2, self.canvas_size[1] // 2)
        width = int(min(self.canvas_size) * size)
        height = int(width * 0.7)
        
        half_width = width // 2
        half_height = height // 2
        
        pt1 = (center[0] - half_width, center[1] - half_height)
        pt2 = (center[0] + half_width, center[1] + half_height)
        
        # 내부를 검은색으로 채움
        cv2.rectangle(canvas, pt1, pt2, (0, 0, 0), -1)
        return canvas
    
    def _draw_square(self, size=0.7):
        canvas = np.ones((*self.canvas_size, 3), dtype=np.uint8) * 255
        center = (self.canvas_size[0] // 2, self.canvas_size[1] // 2)
        side = int(min(self.canvas_size) * size)
        half_side = side // 2
        
        pt1 = (center[0] - half_side, center[1] - half_side)
        pt2 = (center[0] + half_side, center[1] + half_side)
        
        # 내부를 검은색으로 채움
        cv2.rectangle(canvas, pt1, pt2, (0, 0, 0), -1)
        return canvas
    
    def _draw_circle(self, size=0.7):
        canvas = np.ones((*self.canvas_size, 3), dtype=np.uint8) * 255
        center = (self.canvas_size[0] // 2, self.canvas_size[1] // 2)
        radius = int(min(self.canvas_size) * size // 2)
        
        # 내부를 검은색으로 채움
        cv2.circle(canvas, center, radius, (0, 0, 0), -1)
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
        # 내부를 검은색으로 채움
        cv2.fillPoly(canvas, [pts], (0, 0, 0))
        return canvas
    
    def _draw_star(self, size=0.7, points=5):
        canvas = np.ones((*self.canvas_size, 3), dtype=np.uint8) * 255
        center = (self.canvas_size[0] // 2, self.canvas_size[1] // 2)
        radius_outer = int(min(self.canvas_size) * size // 2)
        radius_inner = radius_outer // 2
        
        angles = np.linspace(0, 2*np.pi, points*2, endpoint=False)
        pts = []
        
        for i, angle in enumerate(angles):
            radius = radius_outer if i % 2 == 0 else radius_inner
            x = center[0] + int(radius * np.cos(angle - np.pi/2))
            y = center[1] + int(radius * np.sin(angle - np.pi/2))
            pts.append([x, y])
        
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        # 내부를 검은색으로 채움
        cv2.fillPoly(canvas, [pts], (0, 0, 0))
        return canvas
    
    def generate_shape(self, shape_name):
        """도형 생성"""
        shape_name = shape_name.lower()
        
        # 한국어 형태명 매핑
        if "사각형" in shape_name and "정" in shape_name:
            shape_func = self.shapes["정사각형"]
        elif "사각형" in shape_name or "네모" in shape_name:
            shape_func = self.shapes["사각형"]
        elif "원" in shape_name:
            shape_func = self.shapes["원"]
        elif "삼각형" in shape_name or "세모" in shape_name:
            shape_func = self.shapes["삼각형"]
        elif "별" in shape_name:
            shape_func = self.shapes["별"]
        else:
            # 기본값은 정사각형
            shape_func = self.shapes["정사각형"]
            
        return shape_func()