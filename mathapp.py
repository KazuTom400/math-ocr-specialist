import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from docx import Document
import io, re, os
import numpy as np
from src.loader import RobustLatexOCR

# --- プロダクト・グレードのキャッシュ戦略 ---
@st.cache_resource
def get_ocr_expert():
    # アセットの場所を指定（相対パスで管理）
    asset_dir = os.path.join(os.path.dirname(__file__), "assets")
    return RobustLatexOCR(asset_dir)

st.set_page_config(page_title="MathOCR Specialist", layout="wide")

# --- 専門家の召喚 ---
try:
    ocr_expert = get_ocr_expert()
except Exception as e:
    st.error(f"🚨 システム初期化エラー: {e}")
    st.info("GitHub LFSで .pth ファイルが正しく取得されているか、assets フォルダを確認してください。")
    st.stop()

st.title("🎯 数式ターゲット・スキャナー")

# セッション管理
if 'latex_results' not in st.session_state:
    st.session_state['latex_results'] = []

# --- 物理数学辞書 (35種) ---
PM_BOSS_DICT = {
    "a": [r"a", r"\alpha", r"\mathbf{a}", r"A", r"\mathcal{A}", r"\hat{a}"],
    "b": [r"b", r"\beta", r"B", r"\mathbf{B}"],
    "d": [r"d", r"\delta", r"\Delta", r"\partial", r"\nabla"],
    "e": [r"e", r"E", r"\epsilon", r"\varepsilon"],
    "f": [r"f", r"F", r"\phi", r"\varphi", r"\Phi"],
    "g": [r"g", r"G", r"\gamma", r"\Gamma"],
    "h": [r"h", r"\hbar", r"H", r"\hat{H}", r"\mathcal{H}"],
    "l": [r"l", r"\ell", r"L", r"\lambda", r"\Lambda"],
    "p": [r"p", r"\psi", r"\Psi", r"\rho", r"\phi"],
    "w": [r"w", r"W", r"\omega", r"\Omega"],
    # ... 他、必要に応じて追加
}

# --- 共通ロジック ---
def update_latex(key, target, replacement, n):
    st.session_state[key] = replace_occurrence(st.session_state[key], target, replacement, n)

def replace_occurrence(text, target, replacement, n):
    if target.startswith('\\'):
        return re.sub(re.escape(target) + r'(?![a-zA-Z])', replacement, text)
    pattern = r'(\\[a-zA-Z]+)|(' + re.escape(target) + r')'
    if n == -1:
        return re.sub(pattern, lambda m: m.group(1) if m.group(1) else replacement, text)
    matches = list(re.finditer(pattern, text))
    targets = [m for m in matches if m.group(2)]
    if not targets or n >= len(targets): return text
    m = targets[n]
    return text[:m.start()] + replacement + text[m.end():]

# --- メイン UI ---
uploaded_file = st.sidebar.file_uploader("数式画像を投入", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    img = Image.open(uploaded_file)
    display_width = 1000
    scale = display_width / img.width
    display_height = int(img.height * scale)

    st.subheader("1. 範囲選択")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.2)",
        background_image=img.resize((display_width, display_height)),
        height=display_height, width=display_width,
        drawing_mode="rect", key="math_canvas"
    )

    if st.button("🚀 解析開始"):
        if canvas_result.json_data and canvas_result.json_data["objects"]:
            st.session_state['latex_results'] = [] 
            for i, obj in enumerate(canvas_result.json_data["objects"]):
                c_l, c_t, c_w, c_h = obj["left"], obj["top"], obj["width"], obj["height"]
                if c_w < 0: c_l, c_w = c_l + c_w, abs(c_w)
                if c_h < 0: c_t, c_h = c_t + c_h, abs(c_h)
                cropped = img.crop((int(c_l/scale), int(c_t/scale), int((c_l+c_w)/scale), int((c_t+c_h)/scale)))
                try:
                    latex = ocr_expert.predict(cropped)
                    st.session_state['latex_results'].append({"id": i, "latex": latex, "crop_img": cropped})
                except Exception as e:
                    st.error(f"解析失敗: {e}")
            st.rerun()

    if st.session_state['latex_results']:
        st.markdown("---")
        st.subheader("2. ハイブリッド修正")
        all_final = ""
        for idx, item in enumerate(st.session_state['latex_results']):
            edit_key = f"edit_{idx}"
            if edit_key not in st.session_state: st.session_state[edit_key] = item['latex']
            with st.expander(f"数式 {idx+1}", expanded=True):
                col1, col2 = st.columns([1, 2])
                with col1: st.image(item['crop_img'], use_column_width=True)
                with col2: current = st.text_area("LaTeX編集", key=edit_key, height=100)
                # ( ... 辞書置換ロジック ... )
                st.latex(current)
                all_final += current + "\n\n"

        if st.button("📝 すべてをWordに保存"):
            doc = Document()
            for line in all_final.split('\n'):
                if line.strip(): doc.add_paragraph(line.strip())
            bio = io.BytesIO()
            doc.save(bio)
            st.download_button("📥 Wordファイルをダウンロード", bio.getvalue(), "math_results.docx")