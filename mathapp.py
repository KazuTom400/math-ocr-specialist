import streamlit as st
import os
import io
import base64
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from src.loader import RobustLatexOCR

# --- 1. 画像をBase64に変換（真っ白バグ回避の魔法） ---
def get_image_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- 2. 専門パレットの設定 ---
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

# --- 5. セッション状態の管理 ---
if "latex_res" not in st.session_state:
    st.session_state.latex_res = ""

uploaded_file = st.sidebar.file_uploader("📷 数式画像をアップロード", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img_raw = Image.open(uploaded_file).convert("RGB")
    
    col_img, col_ctrl = st.columns([6, 4])
    
    with col_img:
        st.subheader("📏 解析範囲をマウスで囲んでください")
        
        # 表示サイズ計算
        CANVAS_WIDTH = 750 # 画面に収まりやすい幅
        scale = CANVAS_WIDTH / img_raw.width
        canvas_height = int(img_raw.height * scale)
        img_resized = img_raw.resize((CANVAS_WIDTH, canvas_height), resample=Image.LANCZOS)
        
        # キャンバス設定
        # keyを以前と変えることで、ブラウザのキャッシュバグを強制リセットします
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#e67e22",
            background_image=img_resized,
            update_streamlit=True,
            height=canvas_height,
            width=CANVAS_WIDTH,
            drawing_mode="rect", # 四角形選択を維持
            key="canvas_final_production", 
        )
        st.caption("※マウスでドラッグして数式を囲んでください。")

    with col_ctrl:
        st.subheader("📝 修正 & 専門パレット")
        
        # 選択範囲の処理
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                obj = objects[-1] # 最新の矩形
                l, t = int(obj["left"]/scale), int(obj["top"]/scale)
                w, h = int(obj["width"]/scale), int(obj["height"]/scale)
                
                # 負のサイズやゼロを防ぐガード
                w, h = max(w, 1), max(h, 1)
                crop = img_raw.crop((l, t, l + w, t + h))
                
                # プレビュー表示 (1.29.0互換引数)
                st.image(crop, caption="選択範囲のプレビュー", use_column_width=True)
                
                if st.button("✨ この範囲を解析実行"):
                    with st.spinner("AI解析中..."):
                        res = ocr.predict(crop)
                        st.session_state.latex_res = res.replace("$", "").strip()

        # --- 【あなたの最強機能：修正パレット】 ---
        if st.session_state.latex_res:
            st.divider()
            
            # ルート1: キーボード修正
            st.markdown("**⌨️ ルート1: 文字指定修正**")
            current = st.session_state.latex_res
            c1, c2, c3 = st.columns([1, 2, 1])
            target_idx = c1.number_input("位置", 1, len(current) if len(current)>0 else 1, 1)
            
            # 現在の文字を表示しつつ修正
            idx_zero = target_idx - 1
            char_now = current[idx_zero] if idx_zero < len(current) else ""
            new_val = c2.text_input(f"修正（現在: '{char_now}'）", value=char_now)
            
            if c3.button("適用"):
                l_list = list(current)
                if idx_zero < len(l_list):
                    l_list[idx_zero] = new_val
                    st.session_state.latex_res = "".join(l_list)
                    st.rerun()

            # ルート2: 専門文字パレット
            st.markdown("**🌿 ルート2: 特殊記号パレット**")
            t_greek, t_num = st.tabs(["ギリシャ文字", "数字・演算子"])
            
            with t_greek:
                cols = st.columns(5)
                for i, g in enumerate(GREEK_LETTERS):
                    if cols[i % 5].button(f"\\{g}", key=f"btn_{g}"):
                        st.session_state.latex_res += f" \\{g}"
                        st.rerun()
            
            with t_num:
                cols = st.columns(6)
                for i, k in enumerate(KEYBOARD_CHARS):
                    if cols[i % 6].button(k, key=f"btn_{k}"):
                        st.session_state.latex_res += k
                        st.rerun()

            st.success("現在のLaTeX:")
            st.code(st.session_state.latex_res)
            st.latex(st.session_state.latex_res)

            # 追加: Word保存もここに統合しておきます
            # (以前のcreate_docx関数が必要な場合は適宜追加してください)
else:
    st.info("サイドバーから数式画像をアップロードしてください。")
