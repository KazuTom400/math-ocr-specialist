import os
import yaml
import torch
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from argparse import Namespace
from pix2tex.cli import LatexOCR

@dataclass
class ModelConfig:
    """設定値の型定義とバリデーション"""
    # YAML内でリストとして定義されている項目
    backbone_layers: List[int] = field(default_factory=lambda: [2, 3, 7])
    channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    max_dimensions: List[int] = field(default_factory=lambda: [1024, 512]) # [H, W]
    min_dimensions: List[int] = field(default_factory=lambda: [32, 32])
    
    # スカラ値（デフォルト値はYAMLがない場合のフォールバック）
    temperature: float = 0.00001
    max_seq_len: int = 512
    patch_size: int = 16
    dim: int = 256
    decoder_args: Dict[str, Any] = field(default_factory=lambda: {
        'max_seq_len': 512, 'dim': 256, 'num_layers': 4, 'heads': 8
    })

    @classmethod
    def from_yaml(cls, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # クラス定義にあるキーのみを抽出してマッピング
        valid_args = {k: v for k, v in data.items() if k in cls.__annotations__}
        return cls(**valid_args)

class RobustLatexOCR:
    def __init__(self, asset_path: str):
        self.weights = os.path.join(asset_path, "weights.pth")
        self.resizer = os.path.join(asset_path, "resizer.pth")
        self.config_path = os.path.join(asset_path, "settings.yaml")
        
        # 1. 資産のイミュータブル確認
        for p in [self.weights, self.resizer, self.config_path]:
            if not os.path.exists(p):
                raise RuntimeError(f"Critical Asset Missing: {p}")

        # 2. 設定のロード
        self.config = ModelConfig.from_yaml(self.config_path)

        # 3. 引数のサニタイズ（ここが修正の肝）
        # list型の max_dimensions を、pix2texが期待する int型の max_height/max_width に展開
        max_height = self.config.max_dimensions[0] if isinstance(self.config.max_dimensions, list) else self.config.max_dimensions
        max_width = self.config.max_dimensions[1] if isinstance(self.config.max_dimensions, list) else self.config.max_dimensions
        
        min_height = self.config.min_dimensions[0] if isinstance(self.config.min_dimensions, list) else self.config.min_dimensions
        min_width = self.config.min_dimensions[1] if isinstance(self.config.min_dimensions, list) else self.config.min_dimensions

        # Namespaceの構築（明示的に値を指定して上書き）
        args = Namespace(
            # 必須パス
            checkpoint=self.weights,
            config=self.config_path,
            
            # 動作モード
            no_cuda=True, # 初期化時はCPUで安全に
            no_resize=False,
            
            # 展開したスカラ値を明示的に渡す
            max_height=int(max_height),
            max_width=int(max_width),
            min_height=int(min_height),
            min_width=int(min_width),
            patch_size=int(self.config.patch_size),
            
            # その他の設定を展開
            **asdict(self.config)
        )
        
        print(f"🔧 Initializing LatexOCR with Sanitized Args: max_dims=({args.max_height}, {args.max_width})")
        
        try:
            self.engine = LatexOCR(args)
            # モデルロード後にGPUが使えれば転送（オプション）
            if torch.cuda.is_available():
                self.engine.model.cuda()
        except TypeError as e:
            # 万が一のデバッグ用詳細ログ
            raise RuntimeError(f"Initialization failed due to type mismatch: {e}. Args: {vars(args)}")

    def predict(self, image):
        # pix2texの仕様に合わせてPIL Imageを処理
        try:
            return f"${self.engine(image)}$"
        except Exception as e:
            return f"\\text{{Error in processing: {str(e)}}}"
