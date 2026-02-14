import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List
from argparse import Namespace
from pix2tex.cli import LatexOCR

@dataclass
class ModelConfig:
    """AIの骨格を定義。不足や型エラーをここで事前に防ぐ"""
    backbone_layers: List[int] = field(default_factory=lambda: [2, 3, 7])
    channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    num_layers: int = 4
    max_dimensions: List[int] = field(default_factory=lambda: [1024, 512])
    min_dimensions: List[int] = field(default_factory=lambda: [32, 32])
    max_width: int = 1024
    max_height: int = 512
    max_seq_len: int = 512
    temperature: float = 0.00001
    pad_token: int = 0
    eos_token: int = 1
    bos_token: int = 2
    unk_token: int = 3
    patch_size: int = 16
    emb_dim: int = 256
    dec_layers: int = 4
    nhead: int = 8
    resizer: bool = True
    gray: bool = True
    bit_depth: bool = False

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        # 必要なキーだけを抽出し、型安全なDataclassを生成
        valid_keys = cls.__annotations__.keys()
        return cls(**{k: v for k, v in data.items() if k in valid_keys})

class RobustLatexOCR:
    def __init__(self, asset_dir: str):
        self.weights = os.path.join(asset_dir, "weights.pth")
        self.resizer = os.path.join(asset_dir, "resizer.pth")
        self.config_path = os.path.join(asset_dir, "settings.yaml")

        # 起動前にファイル存在チェック（404やFileNotFoundを事前にトラップ）
        for p in [self.weights, self.resizer, self.config_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"致命的エラー: アセットがありません -> {p}")

        # 設定ファイルのバリデーション
        config_obj = ModelConfig.from_yaml(self.config_path)
        
        # pix2texが期待する形式(Namespace)へブリッジ
        args = Namespace(
            checkpoint=self.weights,
            resizer=self.resizer,
            config=self.config_path,
            no_cuda=True,
            no_gui=True,
            **asdict(config_obj)
        )
        
        # 起動！
        self.model = LatexOCR(args)

    def predict(self, image):
        """推論結果に $ 記号を付けて返す（ビジネスUX対応）"""
        raw_result = self.model(image)
        return f"${raw_result}$"