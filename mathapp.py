import streamlit as st
import os
import io
import re
from PIL import Image
from docx import Document
from docx.shared import Inches
from streamlit_drawable_canvas import st_canvas # これを使います！
from src.loader import RobustLatexOCR

# --- 1. ギリシャ文字・物理定数リスト ---
GREEK_LETTERS = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", 
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", 
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega"
]

# --- 2. 便利関数 ---
def extract_non_keyboard_chars(text):
    found = re.findall(r'\\([a-zA-Z]+)', text)
    return [f"\\{f}" for f in found if f in GREEK_LETTERS]

# --- 3. ページ設定 ---
st.set_page_config(page_title="MathOCR Specialist", layout="wide", page_icon="🎯")
st.title("🎯 MathOCR Specialist")
st.caption("マウスで数式を囲んでスキャンしてください")

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
    
    # 描画キャンバスの横幅を固定してバグを回避
    CANVAS_WIDTH = 700
    scale = CANVAS_WIDTH / img.width
    canvas_height = int(img.height * scale)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 範囲をマウスで囲む")
        
        # 【復活！】四角で囲むキャンバス機能
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # 囲った中身の色
            stroke_width=2,
            stroke_color="#e67e22", # 枠線の色
            background_image=img,
            update_streamlit=True,
            height=canvas_height,
            width=CANVAS_WIDTH,
            drawing_mode="rect", # 四角形モード
            key="canvas",
        )
        
        st.info("💡 数式をマウスでドラッグして囲んでください。")

    with col2:
        st.subheader("🚀 解析・ハイブリッド修正")
        
        # キャンバスからデータを取り出す
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                # 最後に描いた四角形を取得
                obj = objects[-1]
                
                # キャンバス上の座標を元の画像サイズに変換
                real_left = int(obj["left"] / scale)
                real_top = int(obj["top"] / scale)
                real_width = int(obj["width"] / scale)
                real_height = int(obj["height"] / scale)
                
                # クロップ（切り抜き）
                crop = img.crop((real_left, real_top, real_left + real_width, real_top + real_height))
                st.image(crop, caption="ターゲット範囲", use_column_width=True)
                
                if st.button("この範囲を解析する"):
                    with st.spinner("AIが数式を解析中..."):
                        res = ocr.predict(crop)
                        st.session_state.latex_res = res.replace("$", "").strip()

        # --- 修正エリア（あの頃の機能） ---
        if "latex_res" in st.session_state and st.session_state.latex_res:
            current_latex = st.session_state.latex_res
            
            # ルート1: 1文字修正
            st.markdown("**【ルート1】文字・数字の修正**")
            c1, c2 = st.columns([1, 3])
            idx = c1.number_input("何文字目？", 1, len(current_latex), 1)
            new_char = c2.text_input("修正後の文字", value=current_latex[idx-1])
            
            if st.button("ルート1適用"):
                l_list = list(current_latex)
                l_list[idx-1] = new_char
                st.session_state.latex_res = "".join(l_list)
                st.rerun()

            # ルート2: ギリシャ文字
            st.markdown("**【ルート2】ギリシャ文字の確認**")
            found = extract_non_keyboard_chars(current_latex)
            if found:
                st.write("検出された特殊記号:")
                st.write(", ".join(found))
            
            # 結果表示
            st.success("現在の結果")
            st.latex(st.session_state.latex_res)
            st.code(st.session_state.latex_res)

else:
    st.info("左側のサイドバーから画像をアップロードしてください。")
