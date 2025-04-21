import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import cv2
import os
from huggingface_hub import snapshot_download
import time
from tqdm import tqdm

class GenerativeColorizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 더 작고 빠른 모델 사용
        self.model_id = "dreamlike-art/dreamlike-photoreal-2.0"
        
        # 모델 캐시 디렉토리 설정
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 모델 다운로드 및 로딩
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                cache_dir=cache_dir,
                low_cpu_mem_usage=True,
                safety_checker=None
            )
            
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
            else:
                self.pipeline.enable_attention_slicing()
                
        except Exception as e:
            print(f"모델 로딩 중 오류 발생: {str(e)}")
            raise
        
    def generate_texture(self, prompt):
        """
        패션 트렌드에 맞는 색상과 패턴을 생성합니다.
        
        Args:
            prompt (str): 색상과 패턴에 대한 설명
            
        Returns:
            numpy.ndarray: 생성된 이미지 (RGB 형식)
        """
        # 프롬프트에 패션 관련 키워드 추가
        enhanced_prompt = f"fashion design color palette for {prompt}, trending fashion colors, professional fashion design, high quality, detailed, photorealistic"
        
        # 이미지 생성 (더 빠른 설정)
        image = self.pipeline(
            enhanced_prompt,
            num_inference_steps=15,  # 단계 수 감소
            guidance_scale=7.5,
            width=512,
            height=512
        ).images[0]
        
        # PIL 이미지를 numpy 배열로 변환
        return np.array(image)
    
    def apply_texture_to_shape(self, shape_image, texture_image):
        """형태에 텍스처 적용"""
        # 그레이스케일로 변환하여 마스크 생성
        gray = cv2.cvtColor(shape_image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # 마스크 3채널로 변환
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        # 텍스처와 형태 이미지 크기 맞추기
        texture_resized = cv2.resize(texture_image, (shape_image.shape[1], shape_image.shape[0]))
        
        # 마스크를 사용하여 텍스처 적용
        result = np.where(mask_3channel == 0, shape_image, texture_resized)
        
        # 윤곽선 보존
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_with_outline = result.copy()
        cv2.drawContours(result_with_outline, contours, -1, (0, 0, 0), 2)
        
        return result_with_outline