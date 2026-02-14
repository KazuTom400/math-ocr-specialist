import os
import yaml
import torch
from argparse import Namespace
from pix2tex.cli import LatexOCR

class RobustLatexOCR:
    def __init__(self, asset_path: str):
        print("🔍 Starting RobustLatexOCR Initialization...")
        
        self.weights = os.path.join(asset_path, "weights.pth")
        self.resizer = os.path.join(asset_path, "resizer.pth")
        self.raw_config_path = os.path.join(asset_path, "settings.yaml")
        self.clean_config_path = os.path.join(asset_path, "clean_settings.yaml")
        
        # 1. 必須アセットの存在確認
        for p in [self.weights, self.resizer]:
            if not os.path.exists(p):
                raise RuntimeError(f"Critical Asset Missing: {p}")
        
        # settings.yaml は最悪なくても動くようにするので、ここでのチェックは緩める
        if not os.path.exists(self.raw_config_path):
             print(f"⚠️ Warning: {self.raw_config_path} not found. Using internal defaults.")

        # 2. 【核心】「動くことが保証された」完全なデフォルト設定
        # これが Base になります。ユーザー設定ファイルに依存しません。
        full_defaults = {
            # 必須モデルパラメータ (ここが欠けると Munch エラーになる)
            'dim': 256,
            'encoder_structure': 'hybrid',
            'decoder_args': {
                'max_seq_len': 512,
                'dim': 256,
                'num_layers': 4,
                'heads': 8,
                'dropout': 0.1,
            },
            'channels': 1,       # 以前のエラー対策：必ず1
            'patch_size': 16,
            'backbone_layers': [2, 3, 7], # ResNetの構成
            
            # 画像サイズ関連 (デフォルト値)
            'max_height': 192,
            'max_width': 672,
            'min_height': 32,
            'min_width': 32,
            
            # その他トークンIDなど
            'pad_token': 0,
            'bos_token': 1,
            'eos_token': 2,
            'temperature': 0.2,
            'id': None,
            'name': 'math_ocr_model',
            
            # パス情報
            'checkpoint': self.weights,
            'no_cuda': True,
            'no_resize': False,
        }

        # 3. ユーザー設定のロードと「つまみ食い」
        # 元ファイルが壊れていても影響を受けないよう、必要な値だけを取り込む
        user_config = {}
        try:
            if os.path.exists(self.raw_config_path):
                with open(self.raw_config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
                    print("📂 User config loaded.")
        except Exception as e:
            print(f"⚠️ User config could not be loaded ({e}). Using full defaults.")

        # ユーザー設定からサイズ情報だけあれば上書きする (安全なマージ)
        if 'max_dimensions' in user_config and isinstance(user_config['max_dimensions'], list):
            full_defaults['max_height'] = int(user_config['max_dimensions'][0])
            full_defaults['max_width'] = int(user_config['max_dimensions'][1])
            print(f"📏 Updated max dims from user config: {full_defaults['max_height']}x{full_defaults['max_width']}")

        # 4. 「クリーンな設定ファイル」を新規作成
        # ライブラリが後でファイルを読み直しても大丈夫なように、完成品を保存しておく
        try:
            with open(self.clean_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(full_defaults, f)
            print(f"🔧 Generated robust config at: {self.clean_config_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to write clean config: {e}")
        
        # 5. 引数の構築
        # config パスには、今作ったクリーンなファイルを指定
        full_defaults['config'] = self.clean_config_path
        
        # Namespace に変換
        args = Namespace(**full_defaults)
        
        # 最終確認ログ
        print(f"🚀 Initializing LatexOCR with:")
        print(f"   - dim: {getattr(args, 'dim', 'MISSING')}")
        print(f"   - channels: {getattr(args, 'channels', 'MISSING')}")
        
        try:
            self.engine = LatexOCR(args)
            if torch.cuda.is_available():
                self.engine.model.cuda()
            print("✅ Model initialized successfully!")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Model Init Failed: {e}")

    def predict(self, image):
        try:
            return f"${self.engine(image)}$"
        except Exception as e:
            return f"\\text{{Error: {str(e)}}}"
