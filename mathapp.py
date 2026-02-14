import streamlit as st
import os
from PIL import Image
from src.loader import RobustLatexOCR

# --- ページ設定 ---
st.set_page_config(page_title="MathOCR Specialist", layout="wide")

st.title("🎯 MathOCR Specialist (Stable Mode)")

# --- AIエンジンのロード ---
@st.cache_resource
def load_engine():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    asset_dir = os.path.join(base_dir, "assets")
    return RobustLatexOCR(asset_dir)

# ここでエラーが出ないことは既に証明されています！
ocr = load_engine()

# --- メイン UI ---
uploaded_file = st.file_uploader("数式画像をアップロード", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 画像を開く
    img = Image.open(uploaded_file).convert("RGB")
    w, h = img.size
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📏 解析範囲の指定")
        # 1.29.0で確実に動く標準コンポーネントのみを使用
        x_range = st.slider("横の範囲 (左端 - 右端)", 0, w, (0, w), key="x_slider")
        y_range = st.slider("縦の範囲 (上端 - 下端)", 0, h, (0, h), key="y_slider")
        
        # 安全装置: 範囲がゼロにならないようにチェック
        left, right = x_range
        top, bottom = y_range
        
        if right <= left: right = left + 1
        if bottom <= top: bottom = top + 1
        
        # クロップ処理
        crop = img.crop((left, top, right, bottom))
        
        # 【修正点】use_container_width ではなく use_column_width を使用
        # もしくは両方のバージョンで安全なように、引数なしで表示
        st.image(crop, caption="解析対象のプレビュー", use_column_width=True)

    with col2:
        st.subheader("🚀 解析結果")
        if st.button("LaTeXに変換"):
            with st.spinner("数式を解析中..."):
                try:
                    res = ocr.predict(crop)
                    
                    st.success("解析成功！")
                    st.divider()
                    
                    st.markdown("### レンダリング結果")
                    st.latex(res.replace("$", ""))
                    
                    st.markdown("### LaTeXコード (Word等にコピー)")
                    st.code(res, language="latex")
                    
                except Exception as e:
                    st.error(f"解析中にエラーが発生しました: {e}")
else:
    st.info("左側のパネルから画像をアップロードしてください。")
