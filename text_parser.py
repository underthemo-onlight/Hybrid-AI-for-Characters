class TextParser:
    def __init__(self):
        # 색상 사전
        self.color_dict = {
            "빨간": "red", "빨강": "red", "레드": "red",
            "파란": "blue", "파랑": "blue", "블루": "blue", 
            "초록": "green", "그린": "green",
            "노란": "yellow", "노랑": "yellow", "옐로우": "yellow",
            "보라": "purple", "퍼플": "purple",
            "주황": "orange", "오렌지": "orange",
            "검정": "black", "블랙": "black",
            "하양": "white", "화이트": "white",
            "분홍": "pink", "핑크": "pink",
            "갈색": "brown", "브라운": "brown",
            "회색": "gray", "그레이": "gray",
            "연청색": "light blue", "하늘색": "sky blue",
        }
        
        # 형태 사전
        self.shape_dict = {
            "사각형": "rectangle", "네모": "rectangle", "직사각형": "rectangle", "정사각형": "square",
            "원": "circle", "동그라미": "circle",
            "삼각형": "triangle", "세모": "triangle"
        }
    
    def parse_text(self, text):
        """입력 텍스트에서 색상과 형태 추출"""
        text = text.lower()
        
        # 결과 초기화
        result = {
            "color": None,
            "shape": None
        }
        
        # 색상 검색
        for color_kr, color_en in self.color_dict.items():
            if color_kr in text:
                result["color"] = color_en
                break
        
        # 형태 검색
        for shape_kr, shape_en in self.shape_dict.items():
            if shape_kr in text:
                result["shape"] = shape_en
                break
        
        return result