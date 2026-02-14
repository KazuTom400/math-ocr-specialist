import streamlit as st
import os
import io
from PIL import Image
from docx import Document
from docx.shared import Inches
from streamlit_drawable_canvas import st_canvas
from src.loader import RobustLatexOCR

# --- 1. 数学・物理 専門辞書 (復活！) ---
# 認識ミスしやすい記号や、物理で多用するスタイルを自動補正します
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
    """物理辞書を適用し、不必要なスペースを削除して美化する"""
    text = text.replace("$", "").strip()
    for raw, refined in MATH_PHYSICS_DICT.items():
        text = text.replace(raw, refined)
    return text

# --- 2. Word出力機能 (復活！) ---
def create_docx(latex_code, image):
    doc = Document()
    doc.add_heading('MathOCR Analysis Report', 0)
    doc.add_paragraph('解析された数式:')
    doc.add_paragraph(latex_code)
    
    # 画像も添付
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    doc.add_picture(img_byte_arr, width=Inches(4))
    
    target_stream = io.BytesIO()
    doc.save(target_stream)
    return target_stream.getvalue()

# --- 3. ページ設定 ---
st.set_page_config(page_title="MathOCR Specialist", layout="wide", page_icon="🎯")

# カスタムCSSでUIをプロ仕様に
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎯 MathOCR Specialist")
st.caption("数学・物理に特化した高精度数式スキャナー（シニア・エンジニア監修版）")

# --- 4. エンジンロード (安定版 loader を使用) ---
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
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 範囲の指定")
        st.write("数式をドラッグして囲んでください。")
        
        # キャンバス設定（以前の高度な範囲指定を復活）
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  
            stroke_width=2,
            stroke_color="#e67e22",
            background_image=img,
            update_streamlit=True,
            height=img.height * (700 / img.width), # アスペクト比を維持
            width=700,
            drawing_mode="rect",
            key="canvas",
        )

    with col2:
        st.subheader("🚀 解析・出力")
        
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                # 最後に描画された矩形を取得
                obj = objects[-1]
                scale_x = img.width / 700
                scale_y = img.height / (img.height * (700 / img.width))
                
                left = int(obj["left"] * scale_x)
                top = int(obj["top"] * scale_y)
                width = int(obj["width"] * scale_x)
                height = int(obj["height"] * scale_y)
                
                # クロップ
                crop = img.crop((left, top, left + width, top + height))
                st.image(crop, caption="ターゲット範囲", use_container_width=True)
                
                if st.button("数式を解析する"):
                    with st.spinner("物理・数学アルゴリズム適用中..."):
                        try:
                            raw_res = ocr.predict(crop)
                            refined_res = refine_latex(raw_res) # 辞書適用！
                            
                            st.success("解析完了！")
                            st.divider()
                            
                            # レンダリング表示
                            st.latex(refined_res)
                            
                            # LaTeXコード
                            st.code(refined_res, language="latex")
                            
                            # Wordエクスポートボタン
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
                st.warning("解析する範囲をマウスで囲んでください。")
else:
    st.info("左側のサイドバーから数式画像をアップロードしてください。")
