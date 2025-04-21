import streamlit as st
from text_parser import TextParser
from shape_module import ShapeGenerator
from diffusers import StableDiffusionPipeline
import numpy as np
from PIL import Image
import io
import cv2
import torch
import os

# 페이지 설정
st.set_page_config(page_title="하이브리드 도형 생성기", layout="wide")
st.title("하이브리드 도형 생성기")

# 모델 초기화
@st.cache_resource(show_spinner=False)
def load_models():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    parser = TextParser()
    shape_generator = ShapeGenerator()
    
    # Dreamlike Photoreal 2.0 모델 설정
    model_id = "dreamlike-art/dreamlike-photoreal-2.0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 모델 캐시 디렉토리 설정
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    os.makedirs(cache_dir, exist_ok=True)
    
    # 파이프라인 설정
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        cache_dir=cache_dir,
        safety_checker=None
    )
    
    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")
    else:
        pipeline.enable_attention_slicing()
    
    return parser, shape_generator, pipeline

# 모델 로딩
with st.spinner("모델을 로딩 중입니다..."):
    parser, shape_generator, pipeline = load_models()

# 사용자 입력
st.subheader("도형 설정")
user_input = st.text_input("도형, 색상, 소재를 설명해주세요", "연청색의 데님 소재 별")

if st.button("도형 생성하기"):
    with st.spinner("도형 생성 중..."):
        try:
            # 텍스트 파싱
            parsed_result = parser.parse_text(user_input)
            
            # 모듈형 도형 생성
            with st.spinner("도형 생성 중..."):
                shape_image = shape_generator.generate_shape(user_input)
            
            # 생성형 AI로 텍스처 생성
            with st.spinner("텍스처 생성 중..."):
                # 텍스트 파싱 결과에서 색상과 소재 정보 추출
                color = parsed_result.get("color", "")
                
                # 텍스처 생성을 위한 프롬프트 구성
                texture_prompt = (
                    f"extreme close up of {color} fabric texture, "
                    f"detailed material study of {user_input}, "
                    "seamless tileable texture, highly detailed fabric weave pattern, "
                    "realistic fiber details, clear material surface, "
                    "professional studio lighting, centered composition, "
                    "8k texture photography, photorealistic, hyperdetailed"
                )
                
                negative_prompt = (
                    "blur, painting, illustration, drawing, art, cartoon, anime, "
                    "text, letters, numbers, watermark, signature, "
                    "low quality, low resolution, distorted, noisy, "
                    "person, face, hands, abstract, stylized"
                )
                
                # 이미지 생성 파라미터 최적화
                generated_image = pipeline(
                    prompt=texture_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=40,      # 스텝 수 증가
                    guidance_scale=8.5,          # 프롬프트 영향력 증가
                    width=512,
                    height=512,
                ).images[0]
                
                # PIL 이미지를 numpy 배열로 변환
                texture_image = np.array(generated_image)
            
            # 도형에 텍스처 적용
            with st.spinner("텍스처 적용 중..."):
                # 도형 이미지를 마스크로 변환 (흰색 배경, 검은색 도형)
                shape_mask = cv2.cvtColor(shape_image, cv2.COLOR_BGR2GRAY)
                
                # 텍스처 크기 조정
                texture_resized = cv2.resize(texture_image, (shape_image.shape[1], shape_image.shape[0]))
                
                # 마스크를 3채널로 확장
                shape_mask_3channel = np.repeat(shape_mask[:, :, np.newaxis], 3, -1)
                
                # bitwise_and를 사용하여 텍스처 적용
                final_image = cv2.bitwise_and(texture_resized, 255 - shape_mask_3channel)
                
                # 윤곽선 추가
                contours, _ = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(final_image, contours, -1, (0, 0, 0), 2)
            
            # 결과 표시
            st.subheader("생성된 결과")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(shape_image, caption="모듈형 도형", use_container_width=True)
            
            with col2:
                st.image(texture_image, caption="생성형 AI 텍스처", use_container_width=True)
            
            with col3:
                st.image(final_image, caption="최종 결과", use_container_width=True)
            
            # 다운로드 버튼
            img = Image.fromarray(final_image)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="결과 다운로드",
                data=byte_im,
                file_name="hybrid_shape.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# 설명
st.markdown("""
### 사용 방법
도형은 모듈형으로 생성되고, 텍스처는 Dreamlike Photoreal 2.0 모델을 사용하여 생성됩니다.

입력 예시:
- "연청색의 데님 소재 별"
- "빨간색 메탈릭 원"
- "홀로그램 정사각형"
- "글리터 삼각형"

### 지원하는 도형 (모듈형)
- 원
- 사각형
- 정사각형
- 삼각형
- 별

### 텍스처 생성 (생성형 AI)
자유롭게 색상과 소재를 설명해보세요:
- 색상: 빨간색, 파란색, 연청색, 보라색 등
- 소재: 메탈릭, 글리터, 홀로그램, 데님, 가죽 등
""")