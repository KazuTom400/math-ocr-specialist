import streamlit as st
import os
import io
import base64
from PIL import Image
from docx import Document
from streamlit_drawable_canvas import st_canvas
from src.loader import RobustLatexOCR

# --- 1. 【復元】究極物理・数学辞書 (PM_BOSS_DICT) ---
PM_BOSS_DICT = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", 
    "zeta": "ζ", "eta": "η", "theta": "θ", "lambda": "λ", "mu": "μ", 
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ", "omega": "ω",
    "partial": "∂", "nabla": "∇", "infty": "∞", "hbar": "ħ", "times": "×", 
    "div": "÷", "neq": "≠", "approx": "≈", "leq": "≤", "geq": "≥",
    "int": "∫", "sum": "∑", "sqrt": "√", "pm": "±", "mp": "∓",
    "cm^{2}": "cm²", "m/s^{2}": "m/s²", "10^{": "10ⁿ"
}

# --- 2. 通し番号「p」を振るロジック ---
def get_numbered_latex(text):
    """各文字に p(1), p(2)... のインデックスを付与したプレビューを作成"""
    chars = list(text)
    numbered_parts = []
    for i, char in enumerate(chars):
        # LaTeXとしてレンダリング可能な形式で番号を振る
        # アンダーライン付きの p(i) 形式
        numbered_parts.append(f"\\underline{{{char}}}_{{({i+1})}}")
    return "".join(numbered_parts)

# --- 3. ページ設定 ---
st.set_page_config(page_title="MathOCR Specialist", layout="wide", page_icon="🎯")
st.title("🎯 MathOCR Specialist: Hybrid Edition")
st.caption("物理学・数理科学特化：インデックス指定型・高精度修正システム")

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
uploaded_file = st.sidebar.file_uploader("📷 画像をアップロード", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img_raw = Image.open(uploaded_file).convert("RGB")
    
    col_img, col_ctrl = st.columns([6, 4])
    
    with col_img:
        st.subheader("📏 解析範囲の指定")
        DISPLAY_WIDTH = 800
        scale = DISPLAY_WIDTH / img_raw.width
        canvas_height = int(img_raw.height * scale)
        img_resized = img_raw.resize((DISPLAY_WIDTH, canvas_height))
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#e67e22",
            background_image=img_resized,
            update_streamlit=True,
            height=canvas_height,
            width=DISPLAY_WIDTH,
            drawing_mode="rect",
            key="canvas_hybrid_final",
        )

    with col_ctrl:
        st.subheader("🚀 リレー形式：解析と修正")
        
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                obj = objects[-1]
                l, t = int(obj["left"]/scale), int(obj["top"]/scale)
                w, h = int(obj["width"]/scale), int(obj["height"]/scale)
                crop = img_raw.crop((l, t, l + w, t + h))
                st.image(crop, use_column_width=True)
                
                if st.button("✨ 数式を解析"):
                    with st.spinner("AI解析中..."):
                        raw = ocr.predict(crop)
                        st.session_state.latex_res = raw.replace("$", "").strip()

        # --- 魂の「通し番号」修正システム ---
        if st.session_state.latex_res:
            st.divider()
            current = st.session_state.latex_res
            
            # 1. 通し番号付きプレビューの表示
            st.info("💡 修正したい文字の番号（下のp番号）を確認してください")
            numbered_latex = get_numbered_latex(current)
            st.latex(numbered_latex)
            
            # 2. 修正リレー
            st.markdown("### 📝 修正リレー")
            c1, c2 = st.columns([1, 2])
            target_p = c1.number_input("修正するp番号", 1, len(current), 1)
            target_char = current[target_p-1]
            
            # 変換候補の提示
            st.write(f"現在の文字: **{target_char}**")
            
            # 候補ボタン（辞書から生成）
            st.write("変換候補（物理辞書）:")
            cand_cols = st.columns(6)
            for i, (k, v) in enumerate(list(PM_BOSS_DICT.items())[:12]): # 上位を表示
                if cand_cols[i % 6].button(f"{v}", key=f"cand_{i}"):
                    l_list = list(current)
                    l_list[target_p-1] = f"\\{k}" if len(k)>1 else k
                    st.session_state.latex_res = "".join(l_list)
                    st.rerun()

            # 手入力による上書き
            manual_edit = st.text_input("手入力で修正（キーボード文字など）", value=target_char)
            if st.button("手入力適用"):
                l_list = list(current)
                l_list[target_p-1] = manual_edit
                st.session_state.latex_res = "".join(l_list)
                st.rerun()

            st.divider()
            st.success("最終的なLaTeXコード:")
            st.code(st.session_state.latex_res)
            st.latex(st.session_state.latex_res)
else:
    st.info("左側のサイドバーから画像をアップロードしてください。")
