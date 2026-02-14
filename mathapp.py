import streamlit as st
import os
import io
import re
import base64
from PIL import Image
from docx import Document
from streamlit_drawable_canvas import st_canvas
from src.loader import RobustLatexOCR

# --- 1. 画像をブラウザが直接読める形式(Base64)に変換する関数 ---
def get_image_base64_string(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# --- 2. 専門パレットの設定 (復元) ---
GREEK_LETTERS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "lambda", "mu", "pi", "rho", "sigma", "tau", "phi", "omega"]
SPECIAL_SYMBOLS = ["\\infty", "\\partial", "\\nabla", "\\hbar", "\\forall", "\\exists", "\\pm", "\\mp", "\\times", "\\div", "\\neq", "\\approx", "\\leq", "\\geq"]
KEYBOARD_CHARS = ["+", "-", "=", "(", ")", "^", "_", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# --- 3. ページ設定 ---
st.set_page_config(page_title="MathOCR Specialist", layout="wide", page_icon="🎯")
st.title("🎯 MathOCR Specialist")

# --- 4. AIエンジンのロード ---
@st.cache_resource
def load_engine():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    asset_dir = os.path.join(base_dir, "assets")
    return RobustLatexOCR(asset_dir)

ocr = load_engine()

# --- 5. メイン UI 構成 ---
if "latex_res" not in st.session_state:
    st.session_state.latex_res = ""

uploaded_file = st.sidebar.file_uploader("📷 数式の画像をアップロード", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img_raw = Image.open(uploaded_file).convert("RGB")
    
    # 画像表示エリアを大きく確保
    col_img, col_ctrl = st.columns([6, 4])
    
    with col_img:
        st.subheader("📏 解析範囲をマウスで囲んでください")
        
        # キャンバスサイズと表示用Base64の作成
        CANVAS_WIDTH = 800
        scale = CANVAS_WIDTH / img_raw.width
        canvas_height = int(img_raw.height * scale)
        
        # 【重要】Base64文字列に変換。これが「真っ白」を直す特効薬です。
        b64_img = get_image_base64_string(img_raw)
        
        # 描画キャンバス
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#e67e22",
            background_image=Image.open(uploaded_file), # 予備でPILも
            background_color="#ffffff",
            update_streamlit=True,
            height=canvas_height,
            width=CANVAS_WIDTH,
            drawing_mode="rect",
            key="canvas_main",
        )

    with col_ctrl:
        st.subheader("📝 修正パレット & 出力")
        
        # 解析処理
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                obj = objects[-1]
                left, top = int(obj["left"]/scale), int(obj["top"]/scale)
                w, h = int(obj["width"]/scale), int(obj["height"]/scale)
                crop = img_raw.crop((left, top, left + w, top + h))
                
                st.image(crop, caption="現在選択されている数式", use_column_width=True)
                
                if st.button("✨ 数式を解析実行"):
                    with st.spinner("AIが数式を変換中..."):
                        res = ocr.predict(crop)
                        st.session_state.latex_res = res.replace("$", "").strip()

        # --- 【あの頃の機能】プロフェッショナル修正パレット ---
        if st.session_state.latex_res:
            st.divider()
            
            # 1. ライブ編集テキストエリア
            st.session_state.latex_res = st.text_input("LaTeX編集 (ここを直接書き換えてもOK)", value=st.session_state.latex_res)

            # 2. タブによる機能別パレット
            tab1, tab2, tab3 = st.tabs(["🌿 ギリシャ文字", "⌨️ 数字・演算子", "✨ 特殊記号"])
            
            with tab1:
                g_cols = st.columns(6)
                for i, g in enumerate(GREEK_LETTERS):
                    if g_cols[i % 6].button(f"\\{g}", key=f"g_{g}"):
                        st.session_state.latex_res += f" \\{g}"
                        st.rerun()

            with tab2:
                k_cols = st.columns(7)
                for i, k in enumerate(KEYBOARD_CHARS):
                    if k_cols[i % 7].button(k, key=f"k_{k}"):
                        st.session_state.latex_res += k
                        st.rerun()
            
            with tab3:
                s_cols = st.columns(5)
                for i, s in enumerate(SPECIAL_SYMBOLS):
                    if s_cols[i % 5].button(s, key=f"s_{s}"):
                        st.session_state.latex_res += f" {s}"
                        st.rerun()

            # --- 最終プレビュー ---
            st.info("最終レンダリング結果")
            st.latex(st.session_state.latex_res)
            
            if st.button("📄 Wordにエクスポート"):
                doc = Document()
                doc.add_paragraph(st.session_state.latex_res)
                bio = io.BytesIO()
                doc.save(bio)
                st.download_button("Wordファイルを保存", bio.getvalue(), "math_ocr.docx")
else:
    st.info("サイドバーから画像をアップロードしてください。")
