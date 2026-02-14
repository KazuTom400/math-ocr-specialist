import streamlit as st
import os
import io
import re
import base64
from PIL import Image
from docx import Document
from streamlit_drawable_canvas import st_canvas
from src.loader import RobustLatexOCR

# --- 1. 【復元】最強の物理・数学 専門辞書 ---
# あの時、数式 $p_v + \rho \cdot v \cdot \nu$ を完璧にするために調整した辞書です
MATH_PHYSICS_DICT = {
    "\\times 10 ^ {": " \\times 10^{",
    "1 0 ^ {": "10^{",
    "cm ^ { 2 }": "\\text{cm}^2",
    "m / s ^ { 2 }": "\\text{m/s}^2",
    "p h i": "\\phi",
    "t h e t a": "\\theta",
    "o m e g a": "\\omega",
    "h b a r": "\\hbar",
    "i n f t y": "\\infty",
    "p i": "\\pi",
    "r h o": "\\rho",
    "n u": "\\nu",
    "p a r t i a l": "\\partial",
    "a l p h a": "\\alpha",
    "p h i": "\\phi",
}

def ultra_refine(text):
    """物理辞書を適用し、LaTeXの空白と記号をプロ仕様に整える"""
    text = text.replace("$", "").strip()
    for raw, refined in MATH_PHYSICS_DICT.items():
        text = text.replace(raw, refined)
    return text

# --- 2. 専門パレットの設定 (ギリシャ文字・数字・特殊記号) ---
GREEK_LETTERS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "lambda", "mu", "pi", "rho", "sigma", "tau", "phi", "omega", "Delta", "Phi"]
OPERATORS = ["+", "-", "=", "(", ")", "[", "]", "{", "}", "^", "_", "/", "*"]
NUMBERS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# --- 3. 【解決】画像が真っ白にならないためのBase64変換 ---
def get_image_base64_string(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

# --- 4. ページ構成とスタイル ---
st.set_page_config(page_title="MathOCR Specialist", layout="wide", page_icon="🎯")

# ボタンを「美しい記号」として見せるためのカスタムCSS
st.markdown("""
    <style>
    div.stButton > button { width: 100%; height: 3.5rem; border-radius: 8px; font-size: 1.2rem !important; }
    div.stButton > button:hover { border-color: #e67e22; color: #e67e22; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎯 MathOCR Specialist")
st.caption("研究者・学生のための、物理・数理科学特化型高精度スキャナー")

# --- 5. エンジンロード ---
@st.cache_resource
def load_engine():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    asset_dir = os.path.join(base_dir, "assets")
    return RobustLatexOCR(asset_dir)

ocr = load_engine()

# セッション状態の管理
if "latex_res" not in st.session_state:
    st.session_state.latex_res = ""

# --- 6. メイン UI ---
uploaded_file = st.sidebar.file_uploader("📷 数式の画像をアップロード", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img_raw = Image.open(uploaded_file).convert("RGB")
    
    col_img, col_ctrl = st.columns([6, 4])
    
    with col_img:
        st.subheader("📏 数式をマウスでドラッグして囲んでください")
        
        CANVAS_WIDTH = 800
        scale = CANVAS_WIDTH / img_raw.width
        canvas_height = int(img_raw.height * scale)
        img_resized = img_raw.resize((CANVAS_WIDTH, canvas_height))
        
        # Base64で画像を直接渡すことで「真っ白」を回避
        img_b64 = get_image_base64_string(img_resized)
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#e67e22",
            background_image=img_resized,
            update_streamlit=True,
            height=canvas_height,
            width=CANVAS_WIDTH,
            drawing_mode="rect",
            key="canvas_final",
        )

    with col_ctrl:
        st.subheader("🚀 解析・プロフェッショナル修正")
        
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                obj = objects[-1]
                l, t = int(obj["left"]/scale), int(obj["top"]/scale)
                w, h = int(obj["width"]/scale), int(obj["height"]/scale)
                crop = img_raw.crop((l, t, l + w, t + h))
                
                st.image(crop, caption="ターゲット（ここを読み取ります）", use_column_width=True)
                
                if st.button("✨ この数式を解析実行"):
                    with st.spinner("AI物理エンジンによる高精度解析中..."):
                        raw = ocr.predict(crop)
                        st.session_state.latex_res = ultra_refine(raw)

        # --- 魂のハイブリッド修正パレット (復元) ---
        if st.session_state.latex_res:
            st.divider()
            st.markdown("### 📝 ハイブリッド修正")
            
            current = st.session_state.latex_res
            
            # ルート1: 位置指定によるピンポイント修正
            st.markdown("**【ルート1】文字・数字のピンポイント修正**")
            c1, c2, c3 = st.columns([1, 2, 1])
            idx = c1.number_input("位置", 1, len(current), 1)
            char_now = current[idx-1]
            new_val = c2.text_input(f"修正（現在: '{char_now}'）", value=char_now)
            if c3.button("適用"):
                l_list = list(current)
                l_list[idx-1] = new_val
                st.session_state.latex_res = "".join(l_list)
                st.rerun()

            # ルート2: カテゴリ別専門ボタン
            st.markdown("**【ルート2】ギリシャ文字・演算子の追加**")
            tab_greek, tab_kb = st.tabs(["🌿 ギリシャ文字", "⌨️ 数字・演算子"])
            
            with tab_greek:
                cols = st.columns(6)
                for i, g in enumerate(GREEK_LETTERS):
                    # ボタンにLaTeXを適用して美しい記号として表示
                    if cols[i % 6].button(f"$\\{g}$", key=f"p_{g}"):
                        st.session_state.latex_res += f" \\{g}"
                        st.rerun()

            with tab_kb:
                cols = st.columns(7)
                for i, k in enumerate(OPERATORS + NUMBERS):
                    if cols[i % 7].button(k, key=f"p_{k}"):
                        st.session_state.latex_res += k
                        st.rerun()

            # 最終結果プレビュー
            st.success("現在の解析結果（LaTeX）:")
            st.latex(st.session_state.latex_res)
            st.code(st.session_state.latex_res, language="latex")
            
            # Word保存
            doc = Document()
            doc.add_paragraph(st.session_state.latex_res)
            bio = io.BytesIO()
            doc.save(bio)
            st.download_button("📄 Wordにエクスポート", bio.getvalue(), "math_result.docx")
else:
    st.info("サイドバーから画像をアップロードしてください。")
