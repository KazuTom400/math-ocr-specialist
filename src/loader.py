import os
import yaml
import torch
from argparse import Namespace
from pix2tex.cli import LatexOCR

class RobustLatexOCR:
    def __init__(self, asset_path: str):
        print("🔍 Starting RobustLatexOCR Initialization (Final Safe Mode)...")
        
        self.weights = os.path.join(asset_path, "weights.pth")
        self.resizer = os.path.join(asset_path, "resizer.pth")
        self.raw_config_path = os.path.join(asset_path, "settings.yaml")
        self.clean_config_path = os.path.join(asset_path, "clean_settings.yaml")
        
        # 1. 必須アセットの確認
        for p in [self.weights, self.resizer]:
            if not os.path.exists(p):
                raise RuntimeError(f"Critical Asset Missing: {p}")

        # 2. 【過剰防衛】全パラメータ網羅型デフォルト設定
        # pix2texの全バージョンに対応できるよう、エイリアス含めて全て定義する
        full_defaults = {
            # --- 基本構造 ---
            'encoder_structure': 'hybrid',
            'dim': 256,
            'channels': 1,       # 必須: 1 (int)
            'patch_size': 16,
            
            # --- エンコーダー詳細 (ここがエラーの主戦場) ---
            'backbone_layers': [2, 3, 7],
            'encoder_depth': 4,  # 前回のエラー原因
            'num_layers': 4,     # encoder_depthのエイリアスとして使われる可能性への保険
            'heads': 8,          # エンコーダーのヘッド数
            
            # --- デコーダー詳細 ---
            'decoder_args': {
                'max_seq_len': 512,
                'dim': 256,
                'num_layers': 4,
                'heads': 8,
                'dropout': 0.1,
                'ff_dim': 1024,  # 追加: FeedForwardの次元
            },
            
            # --- 画像サイズ (int保証) ---
            'max_height': 192,
            'max_width': 672,
            'min_height': 32,
            'min_width': 32,
            
            # --- トークン・学習設定 (推論でも参照される可能性あり) ---
            'pad_token': 0,
            'bos_token': 1,
            'eos_token': 2,
            'temperature': 0.2,
            'dropout': 0.1,
            'emb_dropout': 0.1,
            'micro_batchsize': -1,
            'batchsize': 10,
            'optimizer': 'AdamW',
            'scheduler': 'OneCycleLR',
            'lr': 0.001,
            'seed': 42,
            'id': None,
            'name': 'math_ocr_model',
            'gpu_devices': [],
            
            # --- システム設定 ---
            'checkpoint': self.weights,
            'no_cuda': True,
            'no_resize': False,
        }

        # 3. ユーザー設定のロード (参考程度)
        user_config = {}
        try:
            if os.path.exists(self.raw_config_path):
                with open(self.raw_config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
                    print("📂 User config loaded for reference.")
        except Exception:
            pass

        # 4. 安全なマージ (本当に安全なキーのみ許可)
        # リスト型や構造を壊す可能性のあるキーは一切取り込まない
        safe_keys = ['temperature', 'patch_size', 'dim', 'encoder_depth', 'heads', 'num_layers']
        for k in safe_keys:
            if k in user_config and isinstance(user_config[k], (int, float)):
                full_defaults[k] = user_config[k]
                
        # decoder_args は辞書として慎重に更新
        if 'decoder_args' in user_config and isinstance(user_config['decoder_args'], dict):
            for k, v in user_config['decoder_args'].items():
                if k in full_defaults['decoder_args'] and isinstance(v, (int, float)):
                    full_defaults['decoder_args'][k] = v

        # サイズ情報のマージ (リスト -> int 変換)
        if 'max_dimensions' in user_config and isinstance(user_config['max_dimensions'], list):
            full_defaults['max_height'] = int(user_config['max_dimensions'][0])
            full_defaults['max_width'] = int(user_config['max_dimensions'][1])

        # 5. クリーンな設定ファイルの保存
        try:
            with open(self.clean_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(full_defaults, f)
            print(f"🔧 Generated robust config at: {self.clean_config_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to write clean config: {e}")
        
        # 6. Namespace生成
        full_defaults['config'] = self.clean_config_path
        args = Namespace(**full_defaults)
        
        # 最終パラメータ確認
        print(f"🚀 Initializing LatexOCR with SAFE DEFAULTS:")
        print(f"   - encoder_depth: {args.encoder_depth}")
        print(f"   - heads: {args.heads}")
        print(f"   - dim: {args.dim}")
        print(f"   - channels: {args.channels} (Must be 1)")
        
        try:
            self.engine = LatexOCR(args)
            if torch.cuda.is_available():
                self.engine.model.cuda()
            print("✅ Model initialized successfully!")
        except Exception as e:
            # エラーが出た場合、どの属性が不足していたかを知るためのトレース
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Model Init Failed: {e}")

    def predict(self, image):
        try:
            return f"${self.engine(image)}$"
        except Exception as e:
            return f"\\text{{Error: {str(e)}}}"
