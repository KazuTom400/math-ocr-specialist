import streamlit as st
import os
import io
from PIL import Image
from docx import Document
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

# --- 2. 専門パレットの設定 ---
GREEK_LETTERS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "lambda", "mu", "pi", "rho", "sigma", "tau", "phi", "omega"]
KEYBOARD_CHARS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "=", "(", ")", "^", "_", "/", "*"]

# --- 3. ページ設定 (絶対にTypeErrorを出さない設定) ---
st.set_page_config(page_title="MathOCR Specialist", layout="wide", page_icon="🎯")
st.title("🎯 MathOCR Specialist")
st.caption("研究・卒論用：絶対安定稼働モード（Canvasライブラリ非依存）")

# --- 4. エンジンロード ---
@st.cache_resource
def load_engine():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    asset_dir = os.path.join(base_dir, "assets")
    return RobustLatexOCR(asset_dir)

ocr = load_engine()

if "latex_res" not in st.session_state:
    st.session_state.latex_res = ""

# --- 5. メイン UI ---
uploaded_file = st.sidebar.file_uploader("📷 数式画像をアップロード", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 画像の読み込み
    img_raw = Image.open(uploaded_file).convert("RGB")
    w, h = img_raw.size
    
    col_img, col_ctrl = st.columns([1, 1])
    
    with col_img:
        st.subheader("📏 解析範囲の指定")
        st.info("スライダーを動かして数式を「ターゲット範囲」に収めてください。")
        
        # 左右と上下の範囲をスライダーで指定（これが一番確実です）
        x_range = st.slider("左右の範囲（X座標）", 0, w, (int(w*0.2), int(w*0.8)))
        y_range = st.slider("上下の範囲（Y座標）", 0, h, (int(h*0.3), int(h*0.7)))
        
        # 切り抜き（ROI）
        left, right = x_range
        top, bottom = y_range
        
        # 1px以上の幅を保証
        if right <= left: right = left + 1
        if bottom <= top: bottom = top + 1
        
        crop = img_raw.crop((left, top, right, bottom))
        
        # 100%確実に表示される st.image
        # use_container_width は使わず、1.29.0互換の引数を使用
        st.image(crop, caption="ターゲット範囲（AIがここを読み取ります）", use_column_width=True)

    with col_ctrl:
        st.subheader("🚀 解析・修正パレット")
        
        if st.button("✨ この範囲を解析実行"):
            with st.spinner("AI物理エンジン起動中..."):
                try:
                    res = ocr.predict(crop)
                    st.session_state.latex_res = refine_latex(res)
                except Exception as e:
                    st.error(f"解析エラー: {e}")

        if st.session_state.latex_res:
            st.divider()
            
            # 手動修正セクション
            current = st.session_state.latex_res
            c1, c2, c3 = st.columns([1, 2, 1])
            t_idx = c1.number_input("位置", 1, len(current) if len(current)>0 else 1, 1)
            new_char = c2.text_input(f"修正（現在: '{current[t_idx-1]}'）", value=current[t_idx-1])
            if c3.button("適用"):
                l_list = list(current)
                l_list[t_idx-1] = new_char
                st.session_state.latex_res = "".join(l_list)
                st.rerun()

            # 専門パレット
            tab_greek, tab_kb = st.tabs(["ギリシャ文字", "数字・演算子"])
            with tab_greek:
                cols = st.columns(5)
                for i, g in enumerate(GREEK_LETTERS):
                    if cols[i % 5].button(f"\\{g}", key=f"p_{g}"):
                        st.session_state.latex_res += f" \\{g}"
                        st.rerun()
            
            with tab_kb:
                cols = st.columns(6)
                for i, k in enumerate(KEYBOARD_CHARS):
                    if cols[i % 6].button(k, key=f"p_{k}"):
                        st.session_state.latex_res += k
                        st.rerun()

            st.success("解析結果（LaTeX）:")
            st.code(st.session_state.latex_res)
            st.latex(st.session_state.latex_res)
else:
    st.info("サイドバーから画像をアップロードしてください。")
