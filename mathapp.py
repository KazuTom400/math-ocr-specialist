import streamlit as st
import os
from PIL import Image
from src.loader import RobustLatexOCR

st.set_page_config(page_title="MathOCR Specialist", layout="wide")
st.title("🎯 MathOCR Specialist (Stable Mode)")

@st.cache_resource
def load_engine():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    asset_dir = os.path.join(base_dir, "assets")
    return RobustLatexOCR(asset_dir)

ocr = load_engine()

# --- メイン UI ---
uploaded_file = st.file_uploader("数式画像をアップロード", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    w, h = img.size
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📏 解析範囲の指定")
        # 標準のスライダーを使って切り抜き範囲を指定。100%確実に動作します。
        x_range = st.slider("横の範囲", 0, w, (0, w))
        y_range = st.slider("縦の範囲", 0, h, (0, h))
        
        # クロップ処理
        crop = img.crop((x_range[0], y_range[0], x_range[1], y_range[1]))
        st.image(crop, caption="解析対象 (この画像がAIに送られます)", use_container_width=True)

    with col2:
        st.subheader("🚀 解析結果")
        if st.button("LaTeXに変換"):
            with st.spinner("解析中..."):
                res = ocr.predict(crop)
                st.latex(res.replace("$", ""))
                st.code(res, language="latex")
                st.success("完了！このコードをコピーして利用してください。")
