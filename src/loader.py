import os
import yaml
import torch
import json
import requests
from argparse import Namespace
from pix2tex.cli import LatexOCR

class RobustLatexOCR:
    def __init__(self, asset_path: str):
        print("🔍 Starting RobustLatexOCR Initialization (Self-Healing Mode)...")
        
        self.asset_path = asset_path
        self.weights = os.path.join(asset_path, "weights.pth")
        self.resizer = os.path.join(asset_path, "resizer.pth")
        self.tokenizer_path = os.path.join(asset_path, "tokenizer.json")
        self.raw_config_path = os.path.join(asset_path, "settings.yaml")
        self.clean_config_path = os.path.join(asset_path, "clean_settings.yaml")
        
        # 1. トークナイザーの自己修復 (これが今回の修正の肝)
        self.ensure_tokenizer()

        # 2. 必須アセットの確認
        for p in [self.weights, self.resizer]:
            if not os.path.exists(p):
                raise RuntimeError(f"Critical Asset Missing: {p}")

        # 3. Tokenizerからnum_tokensを取得
        vocab_size = 8000
        try:
            with open(self.tokenizer_path, 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)
                if 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']:
                    vocab_size = len(tokenizer_data['model']['vocab'])
                    print(f"📊 Vocab size loaded: {vocab_size}")
        except Exception as e:
            print(f"⚠️ Warning: Could not read vocab size ({e}). Using default: {vocab_size}")

        # 4. パラメータ定義
        full_defaults = {
            'num_tokens': vocab_size,
            'max_seq_len': 512,
            'dim': 256,
            'encoder_structure': 'hybrid',
            'decoder_structure': 'transformer',
            'backbone_layers': [2, 3, 7],
            'encoder_depth': 4,
            'channels': 1,
            'patch_size': 16,
            'num_layers': 4,
            'heads': 8,
            'ff_dim': 1024,
            'dropout': 0.1,
            'emb_dropout': 0.1,
            'decoder_args': {
                'attn_on_attn': True,
                'cross_attend': True,
                'ff_glu': True,
                'rel_pos_bias': False,
                'use_scalenorm': False,
            },
            'max_height': 192,
            'max_width': 672,
            'min_height': 32,
            'min_width': 32,
            'pad_token': 0,
            'bos_token': 1,
            'eos_token': 2,
            'unk_token': 3,
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
            'checkpoint': self.weights,
            'tokenizer': self.tokenizer_path,
            'id': None,
            'name': 'math_ocr_model',
            'no_cuda': True,
            'no_resize': False,
            'config': self.clean_config_path,
        }

        # 5. ユーザー設定のロード (参考程度)
        user_config = {}
        try:
            if os.path.exists(self.raw_config_path):
                with open(self.raw_config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
        except Exception:
            pass

        # 6. 安全なマージ
        for k, v in user_config.items():
            if k == 'max_dimensions' and isinstance(v, list):
                full_defaults['max_height'] = int(v[0])
                full_defaults['max_width'] = int(v[1])
            elif k == 'min_dimensions' and isinstance(v, list):
                full_defaults['min_height'] = int(v[0])
                full_defaults['min_width'] = int(v[1])
            elif k in full_defaults and isinstance(v, (int, float, str, bool)):
                full_defaults[k] = v
            elif k == 'decoder_args' and isinstance(v, dict):
                for dk, dv in v.items():
                    if dk not in ['dim', 'heads', 'num_layers', 'ff_dim', 'num_tokens']:
                        full_defaults['decoder_args'][dk] = dv

        # 7. クリーンな設定ファイルの保存
        try:
            with open(self.clean_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(full_defaults, f)
        except Exception as e:
            raise RuntimeError(f"Failed to write clean config: {e}")
        
        # 8. Namespace生成
        args = Namespace(**full_defaults)
        
        print(f"🚀 Initializing LatexOCR...")
        
        try:
            self.engine = LatexOCR(args)
            if torch.cuda.is_available():
                self.engine.model.cuda()
            print("✅ Model initialized successfully!")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Model Init Failed: {e}")

    def ensure_tokenizer(self):
        """
        トークナイザーファイルが壊れている(LFSポインタ)か、存在しない場合に
        公式リポジトリから強制的にダウンロードして上書き修復する。
        """
        url = "https://github.com/lukas-blecher/LaTeX-OCR/raw/main/pix2tex/model/dataset/tokenizer.json"
        
        needs_download = False
        
        if not os.path.exists(self.tokenizer_path):
            print("⚠️ Tokenizer not found. Preparing download...")
            needs_download = True
        else:
            # ファイルはあるが、中身がLFSポインタ(テキスト)かチェック
            try:
                with open(self.tokenizer_path, 'r', encoding='utf-8') as f:
                    content = f.read(100) # 先頭だけ読む
                    if "version https://git-lfs" in content:
                        print("⚠️ Tokenizer is an LFS pointer. Preparing download...")
                        needs_download = True
                    else:
                        # JSONとして正当かチェック
                        f.seek(0)
                        json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Tokenizer JSON is corrupted. Preparing download...")
                needs_download = True
            except Exception:
                needs_download = True

        if needs_download:
            print(f"⬇️ Downloading tokenizer from {url}...")
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                with open(self.tokenizer_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print("✅ Tokenizer downloaded and repaired.")
            except Exception as e:
                # ダウンロードも失敗したら打つ手なしだが、エラーログは出す
                raise RuntimeError(f"Failed to download tokenizer: {e}")

    def predict(self, image):
        try:
            return f"${self.engine(image)}$"
        except Exception as e:
            return f"\\text{{Error: {str(e)}}}"
