import streamlit as st
import os
import io
import json
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from src.loader import RobustLatexOCR

# --- 1. 専門パレットの設定 ---
GREEK_LETTERS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "lambda", "mu", "pi", "rho", "sigma", "tau", "phi", "omega"]
KEYBOARD_CHARS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "=", "(", ")", "^", "_", "/", "*"]

# --- 2. ページ設定 ---
st.set_page_config(page_title="MathOCR Specialist", layout="wide", page_icon="🎯")
st.title("🎯 MathOCR Specialist")

# --- 3. エンジンロード ---
@st.cache_resource
def load_engine():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    asset_dir = os.path.join(base_dir, "assets")
    return RobustLatexOCR(asset_dir)

ocr = load_engine()

# セッション状態の初期化
if "latex_res" not in st.session_state:
    st.session_state.latex_res = ""

uploaded_file = st.sidebar.file_uploader("📷 数式画像をアップロード", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img_raw = Image.open(uploaded_file).convert("RGB")
    
    col_img, col_ctrl = st.columns([6, 4])
    
    with col_img:
        st.subheader("📏 解析範囲をマウスで囲んでください")
        
        # 表示サイズの計算
        CANVAS_WIDTH = 750
        scale = CANVAS_WIDTH / img_raw.width
        canvas_height = int(img_raw.height * scale)
        img_resized = img_raw.resize((CANVAS_WIDTH, canvas_height), resample=Image.LANCZOS)
        
        # 【真っ白バグ回避の決定打】
        # 画像オブジェクトを直接渡さず、一度ファイルに保存して「パス」で渡す
        temp_bg_path = os.path.join("assets", "temp_bg.png")
        img_resized.save(temp_bg_path)
        
        # 描画キャンバス
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#e67e22",
            # 地雷回避：オブジェクト(img_resized)ではなくパス(temp_bg_path)を渡す
            background_image=Image.open(temp_bg_path), 
            update_streamlit=True,
            height=canvas_height,
            width=CANVAS_WIDTH,
            drawing_mode="rect",
            key="canvas_final_fix", 
        )
        st.caption("※マウスでドラッグして数式を囲むと、右側にプレビューが表示されます。")

    with col_ctrl:
        st.subheader("📝 修正 & 専門パレット")
        
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                obj = objects[-1]
                l, t = int(obj["left"]/scale), int(obj["top"]/scale)
                w, h = int(obj["width"]/scale), int(obj["height"]/scale)
                
                # 負のサイズ防止
                w, h = max(w, 1), max(h, 1)
                crop = img_raw.crop((l, t, l + w, t + h))
                
                # 地雷2：use_container_width ではなく use_column_width
                st.image(crop, caption="選択範囲のプレビュー", use_column_width=True)
                
                if st.button("✨ この範囲を解析実行"):
                    with st.spinner("AI解析中..."):
                        res = ocr.predict(crop)
                        st.session_state.latex_res = res.replace("$", "").strip()

        # --- 【あなたの最強機能：修正パレット】 ---
        if st.session_state.latex_res:
            st.divider()
            
            # ルート1: ピンポイント文字修正
            st.markdown("**⌨️ ルート1: 文字指定修正**")
            current = st.session_state.latex_res
            c1, c2, c3 = st.columns([1, 2, 1])
            target_idx = c1.number_input("位置", 1, len(current) if len(current)>0 else 1, 1)
            
            idx_zero = target_idx - 1
            char_now = current[idx_zero] if idx_zero < len(current) else ""
            new_val = c2.text_input(f"修正（現在: '{char_now}'）", value=char_now)
            
            if c3.button("適用"):
                l_list = list(current)
                if idx_zero < len(l_list):
                    l_list[idx_zero] = new_val
                    st.session_state.latex_res = "".join(l_list)
                    st.rerun()

            # ルート2: 専門文字パレット (Tab分け)
            st.markdown("**🌿 ルート2: 特殊記号パレット**")
            tab_greek, tab_kb = st.tabs(["ギリシャ文字", "数字・演算子"])
            
            with tab_greek:
                cols = st.columns(5)
                for i, g in enumerate(GREEK_LETTERS):
                    if cols[i % 5].button(f"\\{g}", key=f"g_{g}"):
                        st.session_state.latex_res += f" \\{g}"
                        st.rerun()
            
            with tab_kb:
                cols = st.columns(6)
                for i, k in enumerate(KEYBOARD_CHARS):
                    if cols[i % 6].button(k, key=f"k_{k}"):
                        st.session_state.latex_res += k
                        st.rerun()

            st.success("現在のLaTeX:")
            st.code(st.session_state.latex_res)
            st.latex(st.session_state.latex_res)
else:
    st.info("左側のサイドバーから数式画像をアップロードしてください。")
