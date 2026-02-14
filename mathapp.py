import streamlit as st
import os
import base64
from io import BytesIO
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from src.loader import RobustLatexOCR

st.set_page_config(page_title="MathOCR Specialist", layout="wide")
st.title("🎯 MathOCR ROI Specialist")

@st.cache_resource
def load_engine():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    asset_dir = os.path.join(base_dir, "assets")
    return RobustLatexOCR(asset_dir)

ocr_engine = load_engine()

# --- 画像をBase64に変換する関数 (これが真っ白回避の切り札) ---
def get_image_base64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

uploaded_file = st.file_uploader("数式画像をアップロード", type=["jpg", "png", "jpeg"])

if uploaded_file:
    raw_image = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = raw_image.size

    # 表示サイズの決定
    max_w = 800
    scale = max_w / orig_w if orig_w > max_w else 1.0
    display_w = int(orig_w * scale)
    display_h = int(orig_h * scale)

    # 表示用画像を生成
    display_img = raw_image.resize((display_w, display_h), resample=Image.LANCZOS)
    
    # 【重要】画像をBase64文字列に変換
    bg_image_data = get_image_base64(display_img)

    st.info(f"💡 数式を囲ってください (Scale: {scale:.2f})")

    # キャンバス
    # background_image に「画像オブジェクト」ではなく「Base64文字列」を渡す
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="#FF4B4B",
        background_image=display_img, # 前提としてオブジェクトも渡すが
        background_label=bg_image_data, # ライブラリによってはここが効く場合がある
        update_streamlit=True,
        height=display_h,
        width=display_w,
        drawing_mode="rect",
        key="super_final_canvas", 
    )

    # 解析実行
    if st.button("🚀 LaTeXに変換"):
        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            obj = canvas_result.json_data["objects"][-1]
            left, top = int(obj["left"] / scale), int(obj["top"] / scale)
            w, h = int(obj["width"] / scale), int(obj["height"] / scale)
            
            cropped_img = raw_image.crop((left, top, left + w, top + h))
            
            with st.spinner("解析中..."):
                try:
                    latex_res = ocr_engine.predict(cropped_img)
                    st.divider()
                    st.subheader("抽出結果")
                    st.latex(latex_res.replace("$", ""))
                    st.code(latex_res, language="latex")
                except Exception as e:
                    st.error(f"解析失敗: {e}")
