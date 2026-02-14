import os
import yaml
import torch
from argparse import Namespace
from pix2tex.cli import LatexOCR

class RobustLatexOCR:
    def __init__(self, asset_path: str):
        self.weights = os.path.join(asset_path, "weights.pth")
        self.resizer = os.path.join(asset_path, "resizer.pth")
        self.config_path = os.path.join(asset_path, "settings.yaml")
        
        # 1. 資産の整合性チェック
        for p in [self.weights, self.resizer, self.config_path]:
            if not os.path.exists(p):
                raise RuntimeError(f"Critical Asset Missing: {p}")

        # 2. YAMLを「辞書」としてロード（ここが正）
        with open(self.config_path, 'r', encoding='utf-8') as f:
            args_dict = yaml.safe_load(f)

        # 3. 汚染源の外科的除去（Surgical Removal）
        # ライブラリが誤って使うリスト型パラメータを完全に消去
        if 'max_dimensions' in args_dict:
            max_dims = args_dict.pop('max_dimensions')
        else:
            max_dims = [1024, 512] # デフォルト

        if 'min_dimensions' in args_dict:
            min_dims = args_dict.pop('min_dimensions')
        else:
            min_dims = [32, 32] # デフォルト

        # 4. 安全なスカラ型として再注入
        # listかintかを判定して格納
        max_h = max_dims[0] if isinstance(max_dims, list) else max_dims
        max_w = max_dims[1] if isinstance(max_dims, list) else max_dims
        min_h = min_dims[0] if isinstance(min_dims, list) else min_dims
        min_w = min_dims[1] if isinstance(min_dims, list) else min_dims

        # 5. 辞書の上書き・統合
        args_dict.update({
            'checkpoint': self.weights,
            # 【重要】 'config' キーはあえて渡さない！
            # 渡すとライブラリがファイルを再読込してしまい、上記のpopが無意味になるため。
            # 'config': self.config_path,  <-- REMOVED
            
            'no_cuda': True,
            'no_resize': False,
            'max_height': int(max_h),
            'max_width': int(max_w),
            'min_height': int(min_h),
            'min_width': int(min_w),
            # patch_sizeがYAMLにない場合の保険
            'patch_size': int(args_dict.get('patch_size', 16)),
        })

        # 6. Namespace化
        args = Namespace(**args_dict)
        
        print(f"🔧 Initializing LatexOCR (Bypass Mode): max_dims=({args.max_height}, {args.max_width})")
        
        try:
            # これでライブラリはメモリ上の args_dict だけを信じるようになる
            self.engine = LatexOCR(args)
            
            if torch.cuda.is_available():
                self.engine.model.cuda()
                
        except Exception as e:
            # エラーの詳細解析用
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Model Init Failed: {e}")

    def predict(self, image):
        try:
            return f"${self.engine(image)}$"
        except Exception as e:
            return f"\\text{{Error: {str(e)}}}"
