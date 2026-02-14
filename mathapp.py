import streamlit as st
import os
import io
import re
import base64
from PIL import Image
from docx import Document
from streamlit_drawable_canvas import st_canvas
from src.loader import RobustLatexOCR

# --- 1. 画像をデジタル文字列(Base64)に変換する「魔法」 ---
def get_canvas_image_b64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    # 文字列としてエンコード
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_b64}"

# --- 2. 専門パレットの設定 (ギリシャ文字 vs キーボード) ---
GREEK_LETTERS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "lambda", "mu", "pi", "rho", "sigma", "tau", "phi", "omega"]
KEYBOARD_CHARS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "=", "(", ")", "^", "_", "/", "*"]

# --- 3. ページ設定 ---
st.set_page_config(page_title="MathOCR Specialist", layout="wide", page_icon="🎯")
st.title("🎯 MathOCR Specialist")

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

uploaded_file = st.sidebar.file_uploader("📷 数式画像をアップロード", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # オリジナル画像の読み込み
    img_raw = Image.open(uploaded_file).convert("RGB")
    
    col_img, col_ctrl = st.columns([6, 4])
    
    with col_img:
        st.subheader("📏 解析範囲をマウスで囲んでください")
        
        # 画面サイズに合わせたリサイズ
        CANVAS_WIDTH = 800
        scale = CANVAS_WIDTH / img_raw.width
        canvas_height = int(img_raw.height * scale)
        img_resized = img_raw.resize((CANVAS_WIDTH, canvas_height))
        
        # 【最重要】Base64文字列を生成（これが真っ白バグの解決策！）
        img_b64_data = get_canvas_image_b64(img_resized)
        
        # 描画キャンバス
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#e67e22",
            # ここでBase64文字列を直接指定することで、Cloud上でも画像が確実に表示されます
            background_image=img_resized, 
            background_color="#ffffff",
            update_streamlit=True,
            height=canvas_height,
            width=CANVAS_WIDTH,
            drawing_mode="rect",
            key="canvas_deployment_stable", # キーを変えてキャッシュを強制リセット
        )
        st.caption("※マウスでドラッグして数式を囲むと、右側にプレビューが表示されます。")

    with col_ctrl:
        st.subheader("📝 修正 & 専門パレット")
        
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                obj = objects[-1]
                left, top = int(obj["left"]/scale), int(obj["top"]/scale)
                w, h = int(obj["width"]/scale), int(obj["height"]/scale)
                crop = img_raw.crop((left, top, left + w, top + h))
                
                st.image(crop, caption="選択範囲のプレビュー", use_column_width=True)
                
                if st.button("✨ この範囲を解析実行"):
                    with st.spinner("AI解析中..."):
                        res = ocr.predict(crop)
                        st.session_state.latex_res = res.replace("$", "").strip()

        # --- 【復活のハイブリッド修正】 ---
        if st.session_state.latex_res:
            st.divider()
            
            # ルート1: ピンポイント文字修正 (Index指定)
            st.markdown("**⌨️ ルート1: キーボード文字修正**")
            current = st.session_state.latex_res
            col_idx, col_val, col_apply = st.columns([1, 2, 1])
            target_idx = col_idx.number_input("何番目？", 1, len(current), 1)
            new_val = col_val.text_input(f"修正（現在: '{current[target_idx-1]}'）", value=current[target_idx-1])
            if col_apply.button("適用"):
                l_list = list(current)
                l_list[target_idx-1] = new_val
                st.session_state.latex_res = "".join(l_list)
                st.rerun()

            # ルート2: 専門文字パレット (Tab分け)
            st.markdown("**🌿 ルート2: 特殊記号パレット**")
            tab_greek, tab_kb = st.tabs(["ギリシャ文字", "数字・演算子"])
            
            with tab_greek:
                g_cols = st.columns(5)
                for i, g in enumerate(GREEK_LETTERS):
                    if g_cols[i % 5].button(f"\\{g}", key=f"g_{g}"):
                        st.session_state.latex_res += f" \\{g}"
                        st.rerun()
            
            with tab_kb:
                k_cols = st.columns(6)
                for i, k in enumerate(KEYBOARD_CHARS):
                    if k_cols[i % 6].button(k, key=f"k_{k}"):
                        st.session_state.latex_res += k
                        st.rerun()

            st.success("現在のLaTeX:")
            st.code(st.session_state.latex_res)
            st.latex(st.session_state.latex_res)
else:
    st.info("左側のサイドバーから数式画像をアップロードしてください。")
    
