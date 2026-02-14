import torch
from pix2tex.cli import LatexOCR
import argparse
import os

class RobustLatexOCR:
    def __init__(self, asset_path):
        # パスを絶対パスに変換して確実に指定する
        checkpoint_path = os.path.join(asset_path, 'weights.pth')
        resizer_path = os.path.join(asset_path, 'resizer.pth')
        config_path = os.path.join(asset_path, 'settings.yaml')

        # 衝突を避けるため、引数オブジェクトを自作する
        args = argparse.Namespace(
            config=config_path,
            checkpoint=checkpoint_path,
            resizer=resizer_path,
            no_cuda=not torch.cuda.is_available(),
            no_gui=True,
            gnorm=False # 重複エラーを避けるためのダミー設定
        )
        
        # モデルを初期化（引数はキーワードではなく、オブジェクトとして渡す）
        self.model = LatexOCR(args)

    def predict(self, image):
        return self.model(image)
