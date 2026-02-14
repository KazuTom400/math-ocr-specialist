import streamlit as st
import os
import io
import re
from PIL import Image
from docx import Document
from docx.shared import Inches
from src.loader import RobustLatexOCR

# --- 1. 定数・辞書設定 ---
GREEK_LETTERS = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", 
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", 
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega"
]

# --- 2. 便利関数 ---
def extract_non_keyboard_chars(text):
    """LaTeXからギリシャ文字などの特殊記号を抽出する"""
    # \alpha などのパターンを抽出
    found = re.findall(r'\\([a-zA-Z]+)', text)
    return [f"\\{f}" for f in found if f in GREEK_LETTERS]

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
st.title("🎯 MathOCR Specialist")
st.caption("数学・物理特化型：ハイブリッド修正システム搭載 (Streamlit 1.29.0 安定版)")

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
        # 地雷回避1: st_canvasを使わず、スライダーで安全に範囲指定
        x_range = st.slider("横の範囲", 0, w, (0, w))
        y_range = st.slider("縦の範囲", 0, h, (0, h))
        
        crop = img.crop((x_range[0], y_range[0], x_range[1], y_range[1]))
        # 地雷回避2: use_column_width=True を使用
        st.image(crop, caption="解析対象", use_column_width=True)
        
        analyze_btn = st.button("🚀 数式を解析する")

    with col2:
        st.subheader("📝 解析・修正エリア")
        
        # セッション状態で結果を保持
        if "latex_res" not in st.session_state:
            st.session_state.latex_res = ""

        if analyze_btn:
            with st.spinner("AIが数式を読み取り中..."):
                try:
                    res = ocr.predict(crop)
                    st.session_state.latex_res = res.replace("$", "").strip()
                except Exception as e:
                    st.error(f"エラー: {e}")

        if st.session_state.latex_res:
            # --- 修正システム：ここが「あの頃の機能」 ---
            current_latex = st.session_state.latex_res
            st.info("解析結果を修正できます")
            
            # ルート1: キーボード文字修正
            st.markdown("**【ルート1】キーボード文字・数字の修正**")
            c1, c2 = st.columns([1, 3])
            idx_to_edit = c1.number_input("何文字目？", 1, len(current_latex) if current_latex else 1, 1)
            new_char = c2.text_input("新しい文字を入力", value=current_latex[idx_to_edit-1] if current_latex else "")
            
            if st.button("ルート1：適用"):
                l_list = list(current_latex)
                l_list[idx_to_edit-1] = new_char
                st.session_state.latex_res = "".join(l_list)
                st.rerun()

            st.divider()

            # ルート2: ギリシャ文字修正ボタン
            st.markdown("**【ルート2】ギリシャ文字の修正・追加**")
            found_greeks = extract_non_keyboard_chars(current_latex)
            if found_greeks:
                st.write("検出された特殊記号（クリックで一括置換・修正）:")
                g_cols = st.columns(len(found_greeks))
                for i, g in enumerate(found_greeks):
                    if g_cols[i].button(g):
                        # ここに特定の修正ロジックを入れることも可能
                        st.toast(f"{g} が選択されました。必要に応じてルート1で修正してください。")

            # 最終結果表示
            st.success("現在のLaTeX結果:")
            st.latex(st.session_state.latex_res)
            st.code(st.session_state.latex_res, language="latex")

            # Word出力
            docx_data = create_docx(st.session_state.latex_res, crop)
            st.download_button(
                "📄 Wordで保存", docx_data, "math_result.docx", 
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
else:
    st.info("サイドバーから画像をアップロードしてください。")
