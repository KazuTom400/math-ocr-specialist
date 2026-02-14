import streamlit as st
import os
import io
import re
import base64
from PIL import Image
from docx import Document
from streamlit_drawable_canvas import st_canvas
from src.loader import RobustLatexOCR

# --- 1. 物理・数学 専門パレット (復元) ---
GREEK_LETTERS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "lambda", "mu", "pi", "rho", "sigma", "tau", "phi", "omega", "Delta", "Omega"]
KEYBOARD_CHARS = ["+", "-", "=", "(", ")", "[", "]", "{", "}", "^", "_", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def get_image_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- 2. ページ設定 (画像表示エリアを最大化) ---
st.set_page_config(page_title="MathOCR Specialist", layout="wide", page_icon="🎯")

# カスタムCSSでボタンと表示をプロ仕様に
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 4px; border: 1px solid #ddd; }
    .greek-btn { background-color: #e3f2fd; }
    .kb-btn { background-color: #f5f5f5; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎯 MathOCR Specialist")

# --- 3. AIエンジンのロード ---
@st.cache_resource
def load_engine():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    asset_dir = os.path.join(base_dir, "assets")
    return RobustLatexOCR(asset_dir)

ocr = load_engine()

# --- 4. メイン UI 構成 ---
if "latex_res" not in st.session_state:
    st.session_state.latex_res = ""

uploaded_file = st.sidebar.file_uploader("📷 画像をアップロード", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 画像処理
    img_raw = Image.open(uploaded_file).convert("RGB")
    
    # 画面の左右比率を「6:4」にして画像表示を優先
    col_img, col_ctrl = st.columns([6, 4])
    
    with col_img:
        st.subheader("📏 数式を囲む（画像優先表示）")
        
        # キャンバス表示の安定化ロジック
        CANVAS_WIDTH = 800 # より大きく表示
        scale = CANVAS_WIDTH / img_raw.width
        canvas_height = int(img_raw.height * scale)
        img_disp = img_raw.resize((CANVAS_WIDTH, canvas_height))
        
        # 描画キャンバス (マウス操作)
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#e67e22",
            background_image=img_disp,
            update_streamlit=True,
            height=canvas_height,
            width=CANVAS_WIDTH,
            drawing_mode="rect",
            key="canvas",
        )

    with col_ctrl:
        st.subheader("🚀 解析 & 修正パレット")
        
        # 1. 解析実行エリア
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                obj = objects[-1]
                # 正確な座標計算
                left, top = int(obj["left"]/scale), int(obj["top"]/scale)
                w, h = int(obj["width"]/scale), int(obj["height"]/scale)
                crop = img_raw.crop((left, top, left + w, top + h))
                
                # 切り取った部分を大きくプレビュー
                st.image(crop, caption="ターゲット", use_column_width=True)
                
                if st.button("✨ 数式を解析する"):
                    with st.spinner("AI解析中..."):
                        res = ocr.predict(crop)
                        st.session_state.latex_res = res.replace("$", "").strip()

        # 2. 【あの頃の機能】ハイブリッド修正パレット (復活)
        if st.session_state.latex_res:
            st.divider()
            st.markdown("### 📝 ハイブリッド修正")
            
            # テキストエリアでの直接編集
            st.session_state.latex_res = st.text_input("LaTeXコード直接編集", value=st.session_state.latex_res)

            # --- カテゴリ別修正パレット ---
            tab_greek, tab_kb = st.tabs(["🌿 ギリシャ文字", "⌨️ キーボード/数字"])
            
            with tab_greek:
                st.write("クリックで末尾に追加:")
                g_cols = st.columns(6)
                for i, g in enumerate(GREEK_LETTERS):
                    if g_cols[i % 6].button(f"\\{g}", key=f"g_{g}"):
                        st.session_state.latex_res += f" \\{g}"
                        st.rerun()

            with tab_kb:
                st.write("クリックで末尾に追加:")
                k_cols = st.columns(7)
                for i, k in enumerate(KEYBOARD_CHARS):
                    if k_cols[i % 7].button(k, key=f"k_{k}"):
                        st.session_state.latex_res += k
                        st.rerun()

            # --- 最終レンダリング結果 ---
            st.info("最終レンダリング")
            st.latex(st.session_state.latex_res)
            
            # Word保存
            if st.button("📄 Wordに書き出す"):
                doc = Document()
                doc.add_paragraph(st.session_state.latex_res)
                bio = io.BytesIO()
                doc.save(bio)
                st.download_button("ダウンロード", bio.getvalue(), "result.docx")

else:
    st.info("サイドバーから数式画像をアップロードしてください。")
