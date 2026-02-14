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
        # 生成する「無害化された設定ファイル」のパス
        self.clean_config_path = os.path.join(asset_path, "clean_settings.yaml")
        
        # 1. 資産の実在確認
        for p in [self.weights, self.resizer, self.raw_config_path]:
            if not os.path.exists(p):
                raise RuntimeError(f"Critical Asset Missing: {p}")

        # 2. 設定のロードと無害化（Sanitization）
        with open(self.raw_config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # リスト型の次元定義を検出し、整数型の幅・高さに変換して上書き
        if 'max_dimensions' in data and isinstance(data['max_dimensions'], list):
            dims = data.pop('max_dimensions') # リストを削除
            data['max_height'] = int(dims[0])
            data['max_width'] = int(dims[1])
        
        if 'min_dimensions' in data and isinstance(data['min_dimensions'], list):
            dims = data.pop('min_dimensions') # リストを削除
            data['min_height'] = int(dims[0])
            data['min_width'] = int(dims[1])

        # 安全策: patch_sizeなどが欠落していないか確認
        if 'patch_size' not in data:
            data['patch_size'] = 16

        # 3. 無害化した設定を新しいファイルに書き出す（これがトロイの木馬）
        with open(self.clean_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f)
        
        print(f"🔧 Generated sanitized config at: {self.clean_config_path}")

        # 4. 引数の構築
        # configには「無害化したファイルのパス」を渡す
        args = Namespace(
            config=self.clean_config_path,
            checkpoint=self.weights,
            no_cuda=True,
            no_resize=False,
            **data # 念のためデータ自体も展開して渡す
        )
        
        print(f"🚀 Initializing LatexOCR with clean config...")
        
        try:
            self.engine = LatexOCR(args)
            
            if torch.cuda.is_available():
                self.engine.model.cuda()
                
        except Exception as e:
            # 万が一のエラー詳細出力
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Model Init Failed: {e}")

    def predict(self, image):
        try:
            return f"${self.engine(image)}$"
        except Exception as e:
            return f"\\text{{Error: {str(e)}}}"
