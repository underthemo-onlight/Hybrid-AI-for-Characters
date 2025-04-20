import streamlit as st
from color_generator import GenerativeColorizer
from texture_module import TextureGenerator
from shape_module import ShapeGenerator
import numpy as np
from PIL import Image
import io

# 페이지 설정
st.set_page_config(page_title="패션 디자인 AI", layout="wide")

# 제목
st.title("패션 디자인 AI")

# 사이드바 설정
st.sidebar.title("설정")

# 모델 초기화
@st.cache_resource
def load_models():
    colorizer = GenerativeColorizer()
    texture_generator = TextureGenerator()
    shape_generator = ShapeGenerator()
    return colorizer, texture_generator, shape_generator

colorizer, texture_generator, shape_generator = load_models()

# 사용자 입력
col1, col2 = st.columns(2)

with col1:
    st.subheader("디자인 설정")
    user_input = st.text_input("의류/액세서리 형태와 색상을 설명해주세요", "연청색 원피스")
    
    # 텍스처 설명 입력
    texture_description = st.text_input("텍스처 설명 (예: 린넨, 실크, 데님, 레이스 등)", "린넨")

with col2:
    st.subheader("블렌딩 설정")
    # 블렌딩 모드 선택
    blend_mode = st.selectbox(
        "블렌딩 모드 선택",
        ["soft_light", "overlay", "multiply", "normal"],
        index=0,
        help="soft_light: 부드러운 의류 텍스처에 적합\noverlay: 강한 대비가 필요한 경우\nmultiply: 어두운 텍스처 효과\nnormal: 기본 적용"
    )

if st.button("디자인 생성하기"):
    with st.spinner("디자인 생성 중..."):
        try:
            # 형태 생성
            shape_image = shape_generator.generate_shape(user_input)
            
            # 색상 생성
            color_image = colorizer.generate_texture(user_input)
            
            # 텍스처 생성
            texture_image = texture_generator.generate_texture(texture_description)
            
            # 텍스처 적용
            final_image = texture_generator.apply_texture(shape_image, texture_image, blend_mode)
            
            # 결과 표시
            st.subheader("생성된 디자인")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(shape_image, caption="기본 형태", use_column_width=True)
            
            with col2:
                st.image(texture_image, caption="생성된 텍스처", use_column_width=True)
            
            with col3:
                st.image(final_image, caption="최종 디자인", use_column_width=True)
                
            # 이미지 다운로드 버튼
            img = Image.fromarray(final_image)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="디자인 다운로드",
                data=byte_im,
                file_name="fashion_design.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")

# 설명
st.markdown("""
### 사용 방법
1. 의류나 액세서리의 형태와 색상을 설명하세요 (예: '연청색 원피스', '검은색 가죽 가방')
2. 원하는 텍스처를 설명하세요 (예: 린넨, 실크, 데님, 레이스, 니트 등)
3. 적절한 블렌딩 모드를 선택하세요
4. '디자인 생성하기' 버튼을 클릭하세요

### 블렌딩 모드 설명
- **soft_light**: 부드러운 의류 텍스처에 적합한 블렌딩
- **overlay**: 강한 대비가 필요한 경우 사용
- **multiply**: 어두운 텍스처 효과를 원할 때 사용
- **normal**: 기본적인 텍스처 적용

### 텍스처 예시
- 린넨: 자연스러운 질감
- 실크: 부드럽고 광택있는 표면
- 데님: 거친 면 소재
- 레이스: 섬세한 패턴
- 니트: 뜨개질 무늬
- 가죽: 질감있는 표면
""")