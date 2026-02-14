import os
import yaml
import torch
from argparse import Namespace
from pix2tex.cli import LatexOCR

class RobustLatexOCR:
    def __init__(self, asset_path: str):
        self.weights = os.path.join(asset_path, "weights.pth")
        self.resizer = os.path.join(asset_path, "resizer.pth")
        self.raw_config_path = os.path.join(asset_path, "settings.yaml")
        self.clean_config_path = os.path.join(asset_path, "clean_settings.yaml")
        
        # 1. 必須ファイルの確認
        for p in [self.weights, self.resizer, self.raw_config_path]:
            if not os.path.exists(p):
                raise RuntimeError(f"Critical Asset Missing: {p}")

        # 2. 設定ファイルのロードとサニタイズ
        with open(self.raw_config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # --- 次元リストのサニタイズ ---
        for key in ['max_dimensions', 'min_dimensions']:
            if key in data and isinstance(data[key], list):
                dims = data.pop(key)
                prefix = key.split('_')[0] # 'max' or 'min'
                data[f'{prefix}_height'] = int(dims[0])
                data[f'{prefix}_width'] = int(dims[1])

        # --- 【真犯人の修正】チャンネル数の強制キャスト ---
        # Conv2dが期待するのはリストではなく整数（通常は1）
        if 'channels' in data and isinstance(data['channels'], list):
            data['channels'] = 1
        elif 'channels' not in data:
            data['channels'] = 1

        # その他の安全確保
        if 'patch_size' not in data:
            data['patch_size'] = 16

        # 3. クリーンな設定をファイルに書き出す
        with open(self.clean_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f)
        
        print(f"🔧 Generated sanitized config at: {self.clean_config_path}")

        # 4. 引数の構築
        args = Namespace(
            config=self.clean_config_path,
            checkpoint=self.weights,
            no_cuda=True,
            no_resize=False,
            **data
        )
        
        print(f"🚀 Initializing LatexOCR with clean config...")
        
        try:
            self.engine = LatexOCR(args)
            if torch.cuda.is_available():
                self.engine.model.cuda()
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Model Init Failed: {e}")

    def predict(self, image):
        try:
            return f"${self.engine(image)}$"
        except Exception as e:
            return f"\\text{{Error: {str(e)}}}"
