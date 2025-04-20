import numpy as np
import cv2
from PIL import Image

class ColorApplier:
    def __init__(self):
        # 기본 RGB 색상 맵
        self.color_map = {
            "red": (255, 0, 0),
            "blue": (0, 0, 255),
            "green": (0, 255, 0),
            "yellow": (255, 255, 0),
            "purple": (128, 0, 128),
            "orange": (255, 165, 0),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "pink": (255, 192, 203),  
            "brown": (165, 42, 42),
            "gray": (128, 128, 128),
            "light blue": (173, 216, 230),
            "sky blue": (135, 206, 235)
        }
    
    def apply_color(self, shape_image, color_name):
        """형태 이미지에 색상 적용"""
        if color_name not in self.color_map:
            # 기본값은 검정
            color_name = "black"
        
        # OpenCV 이미지를 PIL로 변환
        pil_image = Image.fromarray(shape_image)
        
        # 색상 추출
        color_rgb = self.color_map[color_name]
        
        # 이미지 처리 (윤곽선은 그대로 두고 내부만 칠함)
        opencv_image = np.array(pil_image)
        
        # 색상 적용을 위해 마스크 생성 (흰색 부분 검출)
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 각 윤곽선 내부를 채우기
        for contour in contours:
            # 윤곽선 내부를 색상으로 채움
            cv2.drawContours(opencv_image, [contour], 0, color_rgb, -1)
        
        return opencv_image