import torch
from diffusers import StableDiffusionPipeline
import numpy as np
from PIL import Image
import os

class TextureGenerator:
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
        
    def generate_texture(self, texture_description):
        """
        의류 및 액세서리용 텍스처를 생성합니다.
        
        Args:
            texture_description (str): 텍스처 설명 (예: 린넨, 실크, 데님 등)
            
        Returns:
            numpy.ndarray: 생성된 텍스처 이미지 (RGB 형식)
        """
        # 프롬프트에 패션 관련 키워드 추가
        enhanced_prompt = f"close up detailed texture of {texture_description} fabric, fashion textile, material study, high quality, detailed, professional fashion design, photorealistic"
        
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
        
    def apply_texture(self, base_image, texture_image, blend_mode="soft_light"):
        """
        기본 이미지에 텍스처를 적용합니다.
        
        Args:
            base_image (numpy.ndarray): 기본 이미지
            texture_image (numpy.ndarray): 텍스처 이미지
            blend_mode (str): 블렌딩 모드 ("soft_light", "overlay", "multiply", "normal")
            
        Returns:
            numpy.ndarray: 텍스처가 적용된 이미지
        """
        if blend_mode == "soft_light":
            return self._soft_light_blend(base_image, texture_image)
        elif blend_mode == "overlay":
            return self._overlay_blend(base_image, texture_image)
        elif blend_mode == "multiply":
            return self._multiply_blend(base_image, texture_image)
        else:  # normal
            return texture_image
            
    def _soft_light_blend(self, base, texture):
        """부드러운 의류 텍스처에 적합한 블렌딩"""
        base = base.astype(float) / 255.0
        texture = texture.astype(float) / 255.0
        
        # Soft Light 블렌딩 공식 적용
        result = np.where(texture <= 0.5,
                         base - (1 - 2 * texture) * base * (1 - base),
                         base + (2 * texture - 1) * (self._screen(base) - base))
        
        return (result * 255).astype(np.uint8)
        
    def _screen(self, x):
        """Screen 블렌딩을 위한 보조 함수"""
        return 1 - (1 - x) * (1 - x)
        
    def _overlay_blend(self, base, texture):
        """강한 대비가 필요한 경우의 블렌딩"""
        base = base.astype(float) / 255.0
        texture = texture.astype(float) / 255.0
        
        result = np.where(base <= 0.5,
                         2 * base * texture,
                         1 - 2 * (1 - base) * (1 - texture))
        
        return (result * 255).astype(np.uint8)
        
    def _multiply_blend(self, base, texture):
        """어두운 텍스처 효과를 위한 블렌딩"""
        base = base.astype(float) / 255.0
        texture = texture.astype(float) / 255.0
        
        result = base * texture
        
        return (result * 255).astype(np.uint8)