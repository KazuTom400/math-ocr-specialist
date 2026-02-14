import os
import yaml
import torch
import json
from argparse import Namespace
from pix2tex.cli import LatexOCR

class RobustLatexOCR:
    def __init__(self, asset_path: str):
        print("🔍 Starting RobustLatexOCR Initialization (Corrected Final Mode)...")
        
        self.weights = os.path.join(asset_path, "weights.pth")
        self.resizer = os.path.join(asset_path, "resizer.pth")
        self.tokenizer_path = os.path.join(asset_path, "tokenizer.json")
        self.raw_config_path = os.path.join(asset_path, "settings.yaml")
        self.clean_config_path = os.path.join(asset_path, "clean_settings.yaml")
        
        # 1. 必須アセットの確認
        for p in [self.weights, self.resizer]:
            if not os.path.exists(p):
                raise RuntimeError(f"Critical Asset Missing: {p}")

        # 2. Tokenizerからnum_tokens（語彙数）を自動取得
        # これがないとデコーダーの初期化で死にます
        vocab_size = 8000 # 万が一のためのデフォルト値
        if os.path.exists(self.tokenizer_path):
            try:
                with open(self.tokenizer_path, 'r', encoding='utf-8') as f:
                    tokenizer_data = json.load(f)
                    # tokenizer.jsonの構造に合わせてvocabサイズを取得
                    if 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']:
                        vocab_size = len(tokenizer_data['model']['vocab'])
                        print(f"📊 Auto-detected vocab size (num_tokens): {vocab_size}")
            except Exception as e:
                print(f"⚠️ Failed to read tokenizer.json: {e}. Using default: {vocab_size}")
        else:
             print(f"⚠️ Tokenizer not found at {self.tokenizer_path}. Using default vocab size: {vocab_size}")

        # 3. 【真の完全網羅】全パラメータ定義
        full_defaults = {
            # --- 必須モデル構造 ---
            'num_tokens': vocab_size, # 【今回追加】これが欠けていました
            'max_seq_len': 512,
            'dim': 256,
            'encoder_structure': 'hybrid',
            'decoder_structure': 'transformer',
            
            # --- エンコーダー ---
            'backbone_layers': [2, 3, 7],
            'encoder_depth': 4,
            'channels': 1,
            'patch_size': 16,
            
            # --- デコーダー ---
            'num_layers': 4,
            'heads': 8,
            'ff_dim': 1024,
            'dropout': 0.1,
            'emb_dropout': 0.1,
            
            # --- decoder_args (ネスト用: pix2texの実装によってはここを見る) ---
            'decoder_args': {
                'max_seq_len': 512,
                'dim': 256,
                'num_layers': 4,
                'heads': 8,
                'dropout': 0.1,
                'num_tokens': vocab_size, # ここにも念のため
                'ff_dim': 1024,
            },
            
            # --- 画像サイズ (int保証) ---
            'max_height': 192,
            'max_width': 672,
            'min_height': 32,
            'min_width': 32,
            
            # --- トークンID ---
            'pad_token': 0,
            'bos_token': 1,
            'eos_token': 2,
            'unk_token': 3,
            
            # --- 学習・システム設定 ---
            'temperature': 0.2,
            'batchsize': 10,
            'micro_batchsize': -1,
            'optimizer': 'AdamW',
            'scheduler': 'OneCycleLR',
            'lr': 0.001,
            'min_lr': 0.0001,
            'weight_decay': 0.05,
            'seed': 42,
            'epochs': 10,
            'wandb': False,
            'device': 'cpu',
            'gpu_devices': [],
            'sample_freq': 2000,
            'val_freq': 1,
            'log_freq': 100,
            'workers': 1,
            
            # --- システム設定 ---
            'checkpoint': self.weights,
            'tokenizer': self.tokenizer_path,
            'id': None,
            'name': 'math_ocr_model',
            'no_cuda': True,
            'no_resize': False,
            'config': self.clean_config_path,
        }

        # 4. ユーザー設定のロード (参考程度)
        user_config = {}
        try:
            if os.path.exists(self.raw_config_path):
                with open(self.raw_config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
                    print("📂 User config loaded for reference.")
        except Exception:
            pass

        # 5. 安全なマージ
        for k, v in user_config.items():
            # リスト型の寸法指定は展開して取り込む
            if k == 'max_dimensions' and isinstance(v, list):
                full_defaults['max_height'] = int(v[0])
                full_defaults['max_width'] = int(v[1])
            elif k == 'min_dimensions' and isinstance(v, list):
                full_defaults['min_height'] = int(v[0])
                full_defaults['min_width'] = int(v[1])
            # 基本型のみ取り込む
            elif k in full_defaults and isinstance(v, (int, float, str, bool)):
                full_defaults[k] = v
            # decoder_argsのマージ
            elif k == 'decoder_args' and isinstance(v, dict):
                full_defaults['decoder_args'].update(v)

        # 6. クリーンな設定ファイルの保存
        try:
            with open(self.clean_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(full_defaults, f)
            print(f"🔧 Generated robust config at: {self.clean_config_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to write clean config: {e}")
        
        # 7. Namespace生成
        args = Namespace(**full_defaults)
        
        print(f"🚀 Initializing LatexOCR with:")
        print(f"   - num_tokens: {args.num_tokens}") # 確認用ログ
        print(f"   - encoder_depth: {args.encoder_depth}")
        print(f"   - dim: {args.dim}")
        
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
