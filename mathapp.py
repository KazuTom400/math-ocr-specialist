import streamlit as st
import os
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from src.loader import RobustLatexOCR

# --- ページ設定 (ワイドモードを有効化し、左右の余白を減らす) ---
st.set_page_config(page_title="MathOCR Specialist", layout="wide")

st.title("🎯 MathOCR ROI Specialist")

# --- AIエンジンのロード ---
@st.cache_resource
def load_engine():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    asset_dir = os.path.join(base_dir, "assets")
    return RobustLatexOCR(asset_dir)

try:
    ocr_engine = load_engine()
except Exception as e:
    st.error(f"🚨 Engine Error: {e}")
    st.stop()

# --- ファイルアップロード ---
uploaded_file = st.file_uploader("数式画像をアップロード", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. 画像を読み込み
    raw_image = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = raw_image.size

    # 2. 表示サイズを動的に決定 (画面からはみ出さないように)
    # Streamlitのメインカラムの幅に合わせる（最大800px程度）
    max_display_width = 800
    
    if orig_w > max_display_width:
        display_w = max_display_width
        scale = display_w / orig_w
        display_h = int(orig_h * scale)
    else:
        # 画像が小さい場合はそのままのサイズで表示
        display_w = orig_w
        display_h = orig_h
        scale = 1.0

    st.info(f"💡 マウスで数式を囲ってください (表示サイズ: {display_w}x{display_h})")

    # 3. キャンバスの構築 (画像のサイズをそのまま反映)
    # ここで height/width を display_h/display_w に連動させるのが肝です
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="#FF4B4B",
        background_image=raw_image.resize((display_w, display_h)),
        update_streamlit=True,
        height=display_h,   # 画像の高さに自動調節
        width=display_w,    # 画像の幅に自動調節
        drawing_mode="rect",
        key="math_canvas_v3",
    )

    # 4. 解析実行
    if st.button("🚀 LaTeXに変換"):
        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            obj = canvas_result.json_data["objects"][-1]
            
            # 座標を元画像のスケールに復元
            left = int(obj["left"] / scale)
            top = int(obj["top"] / scale)
            w = int(obj["width"] / scale)
            h = int(obj["height"] / scale)
            
            # クロップ
            cropped_img = raw_image.crop((left, top, left + w, top + h))
            
            # 結果表示
            with st.spinner("解析中..."):
                try:
                    latex_res = ocr_engine.predict(cropped_img)
                    st.divider()
                    st.subheader("抽出結果")
                    st.latex(latex_res.replace("$", ""))
                    st.code(latex_res, language="latex")
                except Exception as e:
                    st.error(f"解析失敗: {e}")
        else:
            st.warning("⚠️ 範囲を選択してください")
