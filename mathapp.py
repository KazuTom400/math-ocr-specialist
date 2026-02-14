import streamlit as st
import os
import io
import base64
from PIL import Image
from docx import Document
from docx.shared import Inches
from src.loader import RobustLatexOCR

# --- 1. 物理・数学 専門辞書 ---
MATH_PHYSICS_DICT = {
    "\\times 10 ^ {": " \\times 10^{",
    "cm ^ { 2 }": "\\text{cm}^2",
    "m / s ^ { 2 }": "\\text{m/s}^2",
    "p h i": "\\phi",
    "t h e t a": "\\theta",
    "o m e g a": "\\omega",
    "h b a r": "\\hbar",
    "i n f t y": "\\infty",
    "p i": "\\pi",
}

def refine_latex(text):
    text = text.replace("$", "").strip()
    for raw, refined in MATH_PHYSICS_DICT.items():
        text = text.replace(raw, refined)
    return text

# --- 2. Word出力機能 ---
def create_docx(latex_code, image):
    doc = Document()
    doc.add_heading('MathOCR Analysis Report', 0)
    doc.add_paragraph('解析された数式:')
    doc.add_paragraph(latex_code)
    
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    doc.add_picture(img_byte_arr, width=Inches(4))
    
    target_stream = io.BytesIO()
    doc.save(target_stream)
    return target_stream.getvalue()

# --- 3. ページ設定 ---
st.set_page_config(page_title="MathOCR Specialist", layout="wide", page_icon="🎯")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎯 MathOCR Specialist")
st.caption("物理・数学研究のための高精度ツール")

# --- 4. エンジンロード ---
@st.cache_resource
def load_engine():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    asset_dir = os.path.join(base_dir, "assets")
    return RobustLatexOCR(asset_dir)

ocr = load_engine()

# --- 5. メイン UI ---
uploaded_file = st.sidebar.file_uploader("数式画像をアップロード", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    w, h = img.size
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📏 解析範囲の指定")
        # 安定性を重視し、スライダー方式を採用。これで「真っ白」を100%回避します。
        x_range = st.slider("横の範囲", 0, w, (int(w*0.1), int(w*0.9)))
        y_range = st.slider("縦の範囲", 0, h, (int(h*0.3), int(h*0.7)))
        
        # 安全なクロップ処理
        l, r = x_range
        t, b = y_range
        if r <= l: r = l + 1
        if b <= t: b = t + 1
        
        crop = img.crop((l, t, r, b))
        # 1.29.0互換の引数を使用
        st.image(crop, caption="ターゲット範囲", use_column_width=True)

    with col2:
        st.subheader("🚀 解析結果")
        
        if st.button("数式を解析する"):
            with st.spinner("専門アルゴリズム適用中..."):
                try:
                    raw_res = ocr.predict(crop)
                    refined_res = refine_latex(raw_res)
                    
                    st.success("解析完了！")
                    st.divider()
                    
                    # プレビュー表示
                    st.markdown("##### プレビュー")
                    st.latex(refined_res)
                    
                    # コード表示
                    st.markdown("##### LaTeXコード")
                    st.code(refined_res, language="latex")
                    
                    # Wordダウンロードボタン
                    docx_data = create_docx(refined_res, crop)
                    st.download_button(
                        label="📄 Word形式で保存 (.docx)",
                        data=docx_data,
                        file_name="math_analysis.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                except Exception as e:
                    st.error(f"解析エラー: {e}")
else:
    st.info("左側のサイドバーから画像をアップロードしてください。")
