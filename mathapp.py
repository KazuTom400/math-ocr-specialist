import streamlit as st
import os
import io
import re
import base64
from PIL import Image
from docx import Document
from docx.shared import Inches
from streamlit_drawable_canvas import st_canvas
from src.loader import RobustLatexOCR

# --- 1. 物理・数学専用データ ---
GREEK_LETTERS = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", 
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", 
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"
]

# --- 2. 画像のBase64変換 (これが「真っ白」バグの特効薬！) ---
def get_image_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

# --- 3. ページ設定 ---
st.set_page_config(page_title="MathOCR Specialist", layout="wide", page_icon="🎯")
st.title("🎯 MathOCR Specialist")
st.caption("マウスで数式を選択 ➔ ハイブリッド修正 ➔ Word出力")

# --- 4. エンジンロード ---
@st.cache_resource
def load_engine():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    asset_dir = os.path.join(base_dir, "assets")
    return RobustLatexOCR(asset_dir)

ocr = load_engine()

# --- 5. メイン UI ---
if "latex_res" not in st.session_state:
    st.session_state.latex_res = ""

uploaded_file = st.sidebar.file_uploader("数式画像をアップロード", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 画像の読み込みとリサイズ
    img_raw = Image.open(uploaded_file).convert("RGB")
    
    # キャンバスサイズに合わせてリサイズ（バグ回避のため重要）
    CANVAS_WIDTH = 700
    aspect_ratio = img_raw.height / img_raw.width
    canvas_height = int(CANVAS_WIDTH * aspect_ratio)
    img_resized = img_raw.resize((CANVAS_WIDTH, canvas_height))
    
    # 【解決策】Base64文字列に変換してから渡す
    img_b64 = get_image_base64(img_resized)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 範囲をマウスで囲む")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#e67e22",
            background_image=img_resized, # PILオブジェクトを渡しつつ
            update_streamlit=True,
            height=canvas_height,
            width=CANVAS_WIDTH,
            drawing_mode="rect",
            key="canvas",
        )
        st.info("💡 数式をマウスでドラッグして囲んでください。")

    with col2:
        st.subheader("🚀 解析結果と修正")
        
        # クロップ処理と解析
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                obj = objects[-1]
                # 元の画像サイズに対する比率でクロップ範囲を計算
                scale = img_raw.width / CANVAS_WIDTH
                left = int(obj["left"] * scale)
                top = int(obj["top"] * scale)
                w = int(obj["width"] * scale)
                h = int(obj["height"] * scale)
                
                crop = img_raw.crop((left, top, left + w, top + h))
                st.image(crop, caption="ターゲット範囲", use_column_width=True)
                
                if st.button("数式を解析"):
                    with st.spinner("AIが読み取り中..."):
                        res = ocr.predict(crop)
                        st.session_state.latex_res = res.replace("$", "").strip()

        # --- ハイブリッド修正システム (ここが魂！) ---
        if st.session_state.latex_res:
            current = st.session_state.latex_res
            
            st.markdown("---")
            # 【ルート1】キーボード文字修正（インデックス指定）
            st.markdown("**⌨️ ルート1：キーボード文字の修正**")
            cols = st.columns([1, 2, 1])
            idx = cols[0].number_input("何番目？", 1, len(current), 1)
            char_to_edit = current[idx-1]
            new_char = cols[1].text_input(f"修正（現在: '{char_to_edit}'）", value=char_to_edit)
            
            if cols[2].button("適用"):
                l_list = list(current)
                l_list[idx-1] = new_char
                st.session_state.latex_res = "".join(l_list)
                st.rerun()

            # 【ルート2】ギリシャ文字クイック修正
            st.markdown("**🌿 ルート2：ギリシャ文字・特殊記号**")
            # 頻出するギリシャ文字をボタンで並べる
            greek_cols = st.columns(6)
            for i, g in enumerate(["alpha", "beta", "gamma", "theta", "pi", "phi"]):
                if greek_cols[i].button(f"\\{g}"):
                    st.session_state.latex_res += f" \\{g}"
                    st.rerun()

            # 結果のプレビュー
            st.success("現在のLaTeXコード:")
            st.code(st.session_state.latex_res, language="latex")
            st.latex(st.session_state.latex_res)

            # Word出力
            doc = Document()
            doc.add_paragraph(st.session_state.latex_res)
            target_stream = io.BytesIO()
            doc.save(target_stream)
            st.download_button("📄 Word保存", target_stream.getvalue(), "result.docx")

else:
    st.info("サイドバーから画像をアップロードしてください。")
