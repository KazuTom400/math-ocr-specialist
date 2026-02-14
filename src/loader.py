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

        # 3. 引数の完全サニタイズ（ここが決定的な修正点）
        args_dict = asdict(self.config)

        # リスト型の値を退避
        max_dims = args_dict.pop('max_dimensions', [1024, 512]) # popして辞書から消去
        min_dims = args_dict.pop('min_dimensions', [32, 32])    # popして辞書から消去

        # 安全にスカラ値を取得
        max_h = max_dims[0] if isinstance(max_dims, list) else max_dims
        max_w = max_dims[1] if isinstance(max_dims, list) else max_dims
        min_h = min_dims[0] if isinstance(min_dims, list) else min_dims
        min_w = min_dims[1] if isinstance(min_dims, list) else min_dims

        # 辞書に必要なキーを追加・上書き
        args_dict.update({
            'checkpoint': self.weights,
            'config': self.config_path,
            'no_cuda': True,
            'no_resize': False,
            'max_height': int(max_h),
            'max_width': int(max_w),
            'min_height': int(min_h),
            'min_width': int(min_w),
            'patch_size': int(self.config.patch_size),
        })

        # Namespaceの構築
        args = Namespace(**args_dict)
        
        print(f"🔧 Initializing LatexOCR with Sanitized Args: max_dims=({args.max_height}, {args.max_width})")
        # 念のための確認ログ：危険なキーが含まれていないか
        if hasattr(args, 'max_dimensions'):
            print("⚠️ Warning: max_dimensions still exists in args!")

        try:
            self.engine = LatexOCR(args)
            if torch.cuda.is_available():
                self.engine.model.cuda()
        except TypeError as e:
            # エラー発生時の引数ダンプ（デバッグ用）
            raise RuntimeError(f"Initialization failed: {e}. Keys provided: {list(args_dict.keys())}")

    def predict(self, image):
        try:
            return f"${self.engine(image)}$"
        except Exception as e:
            return f"\\text{{Error in processing: {str(e)}}}"
