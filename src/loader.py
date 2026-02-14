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
    
    # スカラ値
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

        # 3. 引数のサニタイズと辞書構築
        # まずConfigを辞書化（ベースとなる設定）
        args_dict = asdict(self.config)

        # list型の max_dimensions を展開してスカラ値を取得
        max_dims = self.config.max_dimensions
        min_dims = self.config.min_dimensions
        
        # リストかスカラかを判定して安全に取得
        max_h = max_dims[0] if isinstance(max_dims, list) else max_dims
        max_w = max_dims[1] if isinstance(max_dims, list) else max_dims
        min_h = min_dims[0] if isinstance(min_dims, list) else min_dims
        min_w = min_dims[1] if isinstance(min_dims, list) else min_dims

        # 辞書を更新（ここで重複キーは上書きされるためエラーにならない）
        # pix2texが必要とするキーを明示的にセット
        args_dict.update({
            'checkpoint': self.weights,
            'config': self.config_path,
            'no_cuda': True, # 初期化時はCPUで安全に
            'no_resize': False,
            'max_height': int(max_h),
            'max_width': int(max_w),
            'min_height': int(min_h),
            'min_width': int(min_w),
            'patch_size': int(self.config.patch_size), # 型保証のため再設定
        })

        # Namespaceの構築（辞書をアンパックして渡す）
        args = Namespace(**args_dict)
        
        print(f"🔧 Initializing LatexOCR with Sanitized Args: max_dims=({args.max_height}, {args.max_width})")
        
        try:
            self.engine = LatexOCR(args)
            # モデルロード後にGPUが使えれば転送
            if torch.cuda.is_available():
                self.engine.model.cuda()
        except TypeError as e:
            raise RuntimeError(f"Initialization failed due to type mismatch: {e}. Args keys: {list(args_dict.keys())}")

    def predict(self, image):
        try:
            return f"${self.engine(image)}$"
        except Exception as e:
            return f"\\text{{Error in processing: {str(e)}}}"
