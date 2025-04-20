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
    def __init__(self, max_retries=3):
        # 모델 ID 설정 - 4GB 내외의 모델 사용
        self.model_id = "stabilityai/stable-diffusion-2-1"
        self.max_retries = max_retries
        
        try:
            # 모델 캐시 디렉토리 설정
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            os.makedirs(cache_dir, exist_ok=True)
            
            # 모델 다운로드 (재시도 로직 포함)
            for attempt in range(self.max_retries):
                try:
                    print(f"모델 다운로드 시도 중... (시도 {attempt + 1}/{self.max_retries})")
                    snapshot_download(
                        repo_id=self.model_id,
                        cache_dir=cache_dir,
                        local_files_only=False,
                        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.safetensors"]  # 불필요한 파일 제외
                    )
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    print(f"다운로드 실패: {str(e)}")
                    print("5초 후 재시도...")
                    time.sleep(5)
            
            # 파이프라인 초기화
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                cache_dir=cache_dir,
                low_cpu_mem_usage=True,
                safety_checker=None
            )
            
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
            else:
                self.pipe.enable_attention_slicing()
                
        except Exception as e:
            print(f"모델 로딩 중 오류 발생: {str(e)}")
            raise
        
    def generate_texture(self, color_description, size=(512, 512)):
        """텍스트 설명을 기반으로 텍스처/색상 생성"""
        # 프롬프트 강화 - 텍스처/색상 관련 지시어 추가
        prompt = f"A flat texture of {color_description} color, simple, minimalist"
        
        # 이미지 생성
        with torch.no_grad():  # 메모리 절약
            image = self.pipe(
                prompt, 
                height=size[0],
                width=size[1],
                num_inference_steps=20  # 기본값 사용
            ).images[0]
        
        # PIL에서 numpy 배열로 변환
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