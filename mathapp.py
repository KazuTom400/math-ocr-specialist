import streamlit as st
import os
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from src.loader import RobustLatexOCR

# --- ページ設定 ---
st.set_page_config(page_title="MathOCR Specialist", layout="centered")

# --- CSSで見た目を調整 ---
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        text-align: center;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

st.title("MathOCR ROI Specialist")
st.markdown("画像をアップロードし、**数式部分をマウスで囲って**ください。")

# --- AIエンジンのロード（キャッシュ化） ---
@st.cache_resource
def load_engine():
    # パスを絶対パスで解決
    base_dir = os.path.dirname(os.path.abspath(__file__))
    asset_dir = os.path.join(base_dir, "assets")
    return RobustLatexOCR(asset_dir)

try:
    ocr_engine = load_engine()
    st.success("✅ AI Engine Loaded Successfully")
except Exception as e:
    st.error(f"🚨 Engine Initialization Failed: {e}")
    st.stop()

# --- 画像アップロード ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. 画像を開く
    raw_image = Image.open(uploaded_file).convert("RGB")
    original_w, original_h = raw_image.size

    # 2. 表示サイズを計算 (レスポンシブ対応)
    # キャンバスの幅を700pxに固定し、高さを比率に合わせて自動計算
    CANVAS_WIDTH = 700
    scale_factor = CANVAS_WIDTH / original_w
    canvas_height = int(original_h * scale_factor)
    
    # 表示用にリサイズした画像を作成
    display_image = raw_image.resize((CANVAS_WIDTH, canvas_height))

    # 3. キャンバスの表示
    # ユーザーは「縮小された画像」の上で操作する
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # 選択範囲の色
        stroke_width=2,
        stroke_color="#FF4B4B",
        background_image=display_image,
        update_streamlit=True,
        height=canvas_height,
        width=CANVAS_WIDTH,
        drawing_mode="rect",  # 四角形選択モード
        key="canvas",
    )

    # 4. 解析実行ボタン
    if st.button("🚀 Convert to LaTeX"):
        # 選択範囲があるかチェック
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            
            if len(objects) > 0:
                # 最新のボックスを取得
                obj = objects[-1]
                
                # 5. 座標の逆変換 (重要！)
                # 表示画面(700px)での座標を、元の高画質画像の座標に戻す
                left = int(obj["left"] / scale_factor)
                top = int(obj["top"] / scale_factor)
                width = int(obj["width"] / scale_factor)
                height = int(obj["height"] / scale_factor)
                
                # クロップ（元画像から切り抜き）
                cropped_img = raw_image.crop((left, top, left + width, top + height))
                
                # 確認用に切り抜いた画像を表示（サイドバーなど）
                with st.expander("Processing Crop..."):
                    st.image(cropped_img, caption="AI Input High-Res Crop")

                # AI推論実行
                with st.spinner("Analyzing math formula..."):
                    try:
                        latex_code = ocr_engine.predict(cropped_img)
                        
                        st.divider()
                        st.subheader("Result")
                        # LaTeXとしてレンダリング
                        st.latex(latex_code.replace("$", ""))
                        # コピー用コードブロック
                        st.code(latex_code, language="latex")
                        
                    except Exception as e:
                        st.error(f"Prediction Error: {e}")
            else:
                st.warning("⚠️ Please draw a box around the formula first.")
