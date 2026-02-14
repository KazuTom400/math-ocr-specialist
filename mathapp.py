import streamlit as st
import os
import io
import base64
from PIL import Image
from docx import Document
from streamlit_drawable_canvas import st_canvas
from src.loader import RobustLatexOCR

# --- 1. 【復元】超・強力 物理数学補正アルゴリズム ---
# OCRが間違えやすい物理定数や単位の「揺れ」を完全に修正します
PHYSICS_AUTO_FIX = {
    "\\times 10 ^ {": " \\times 10^{",
    "1 0 ^ {": "10^{",
    "cm ^ { 2 }": "\\text{cm}^2",
    "m / s ^ { 2 }": "\\text{m/s}^2",
    "k g": "\\text{kg}",
    "h b a r": "\\hbar",
    "o m e g a": "\\omega",
    "p h i": "\\phi",
    "t h e t a": "\\theta",
    "d e l t a": "\\delta",
    "D e l t a": "\\Delta",
    "p i": "\\pi",
    "i n f t y": "\\infty",
}

def ultra_refine(text):
    text = text.replace("$", "").strip()
    for raw, fix in PHYSICS_AUTO_FIX.items():
        text = text.replace(raw, fix)
    # 不自然な空白を物理学的に正しい間隔に調整
    return text.replace(" ", " ").replace("  ", " ")

# --- 2. 【復元】プロ仕様パレット定義 ---
GREEKS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "lambda", "mu", "pi", "rho", "sigma", "tau", "phi", "chi", "psi", "omega"]
SPECIALS = ["\\hbar", "\\partial", "\\nabla", "\\infty", "\\int", "\\sum", "\\pm", "\\times", "\\div", "\\neq", "\\approx", "\\leq", "\\geq"]
OPERATORS = ["+", "-", "=", "(", ")", "[", "]", "{", "}", "^", "_", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# --- 3. 【解決】画像真っ白バグを封じるBase64変換 ---
def get_b64_image(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

# --- 4. ページ構成 ---
st.set_page_config(page_title="MathOCR Specialist", layout="wide", page_icon="🎯")

# ボタンを美しくレンダリングするためのCSS
st.markdown("""
    <style>
    div.stButton > button { width: 100%; font-size: 1.2rem !important; height: 3rem; border-radius: 8px; border: 1px solid #ddd; transition: 0.3s; }
    div.stButton > button:hover { border-color: #007bff; color: #007bff; background: #f0f7ff; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #f8f9fa; border-radius: 4px 4px 0 0; padding: 10px 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎯 MathOCR Specialist")
st.caption("シニア・エンジニア監修：物理学・数理科学特化型解析システム")

# --- 5. エンジンロード ---
@st.cache_resource
def load_engine():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    asset_dir = os.path.join(base_dir, "assets")
    return RobustLatexOCR(asset_dir)

ocr = load_engine()

if "latex_res" not in st.session_state:
    st.session_state.latex_res = ""

# --- 6. メイン UI ---
uploaded_file = st.sidebar.file_uploader("📷 数式画像をアップロード", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img_raw = Image.open(uploaded_file).convert("RGB")
    
    col_img, col_ctrl = st.columns([6, 4]) # 画像表示を大きく確保
    
    with col_img:
        st.subheader("📏 直感的な範囲指定")
        
        # キャンバスサイズの最適化
        DISPLAY_WIDTH = 800
        scale = DISPLAY_WIDTH / img_raw.width
        display_height = int(img_raw.height * scale)
        img_resized = img_raw.resize((DISPLAY_WIDTH, display_height))
        
        # 【最重要】Base64で画像をキャンバスに直接埋め込む
        b64_data = get_b64_image(img_resized)
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#ff8c00",
            background_image=img_resized,
            update_streamlit=True,
            height=display_height,
            width=DISPLAY_WIDTH,
            drawing_mode="rect",
            key="pro_canvas",
        )
        st.info("💡 マウスで数式を囲むと、右側に解析準備が整います。")

    with col_ctrl:
        st.subheader("🚀 解析・プロフェッショナル修正")
        
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                obj = objects[-1]
                # 座標を元画像に引き戻す
                l, t = int(obj["left"]/scale), int(obj["top"]/scale)
                w, h = int(obj["width"]/scale), int(obj["height"]/scale)
                crop = img_raw.crop((l, t, l + w, t + h))
                
                st.image(crop, caption="ターゲット（解析対象）", use_column_width=True)
                
                if st.button("✨ この数式を解析する"):
                    with st.spinner("AI物理エンジンによる高精度解析中..."):
                        raw_res = ocr.predict(crop)
                        st.session_state.latex_res = ultra_refine(raw_res)

        # --- プロ仕様パレット (復活) ---
        if st.session_state.latex_res:
            st.divider()
            # ライブ編集
            st.session_state.latex_res = st.text_input("📝 LaTeX編集エリア", value=st.session_state.latex_res)
            
            # タブ分けされた専門ボタン
            tab1, tab2, tab3 = st.tabs(["🌿 ギリシャ文字", "⌨️ 数字・演算子", "⚛️ 物理・特殊記号"])
            
            with tab1:
                cols = st.columns(6)
                for i, g in enumerate(GREEKS):
                    # ボタンに数式をレンダリングしてプロ仕様に
                    if cols[i % 6].button(f"$\\{g}$", key=f"btn_{g}"):
                        st.session_state.latex_res += f" \\{g}"
                        st.rerun()

            with tab2:
                cols = st.columns(7)
                for i, o in enumerate(OPERATORS):
                    if cols[i % 7].button(o, key=f"btn_{o}"):
                        st.session_state.latex_res += o
                        st.rerun()
                        
            with tab3:
                cols = st.columns(5)
                for i, s in enumerate(SPECIALS):
                    if cols[i % 5].button(f"${s}$", key=f"btn_{i}"):
                        st.session_state.latex_res += f" {s}"
                        st.rerun()

            # 最終プレビュー
            st.success("解析結果（数式プレビュー）:")
            st.latex(st.session_state.latex_res)
            st.code(st.session_state.latex_res, language="latex")
            
            # Word保存機能も復活
            doc = Document()
            doc.add_paragraph(st.session_state.latex_res)
            bio = io.BytesIO()
            doc.save(bio)
            st.download_button("📄 Wordにエクスポート", bio.getvalue(), "math_report.docx")

else:
    st.info("サイドバーから画像をアップロードしてください。")
