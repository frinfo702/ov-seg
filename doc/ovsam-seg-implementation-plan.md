# OVSAM-Seg 実装計画

## モチベーション

OV-Seg (Liang et al., 2023) の追試と定性分析により、以下のボトルネックが判明した:

1. **MaskFormer のマスク切り出し品質**: クラス非依存のマスクプロポーザルとして MaskFormer を使っているが、マスクの形状精度が不十分。クエリによらずマスク形状が変化しない（例: "耳" でクエリしても "人" 全体のマスクしか得られない）
2. **CLIP の領域分類**: fine-grained / 属性認識に弱い（犬種の区別、色の識別など）

本計画では (1) に対して **MaskFormer を SAM3 で置き換える** ことでアプローチする。

## 現状のアーキテクチャ (OV-Seg)

```bash
Input Image
    │
    ▼
Backbone (Swin-B / ResNet-101c)          ◀── パラメータ多い、重い
    │
    ▼
Pixel Decoder (FPN)
    │
    ▼
Transformer Decoder (6層, 100 queries)
    │
    ├──→ pred_masks  [100, H/4, W/4]     ◀── マスク品質がイマイチ
    │
    └──→ mask_embeddings  [100, 768]
            │
            ▼
         cosine similarity with CLIP text
            │
            ▼
         mask_cls  [100, num_classes]     ◀── MaskFormer のスコア
            │
         ┌──┴──┐
         │     │
    [λ=0.7]  [λ=0]
         │     │
         ▼     ▼
    mask_cls  clip_cls  (幾何平均 or 単独)
         │     │
         └──┬──┘
            ▼
    semseg = einsum("qc,qhw→chw", mask_cls, mask_pred)
```

**問題点:**

- backbone → pixel decoder → transformer decoder のパイプラインが重い（Swin-B + 6層Transformer）
- マスク品質が MaskFormer の学習に依存（OV-Seg では学習済み MaskFormer を freeze）
- mask_embeddings と CLIP の線形和 fusion はアーキテクチャを複雑にしている

## 新しいアーキテクチャ (OVSAM-Seg)

```bash
Input Image
    │
    ├──→ SAM3
    │       │
    │       text_prompts = dataset_classes  (e.g., ADE20K-150 classes)
    │       │
    │       ▼
    │    SAM3 detector forward
    │       │
    │       ▼
    │    pred_masks  [N, H_feat, W_feat]   ◀── SAM3 の高品質マスク
    │    pred_boxes  [N, 4]
    │    pred_logits [N]  (confidence)
    │    presence_logit  [1]
    │       │
    │       ▼
    │    Post-process:
    │      - mask_threshold  (e.g., 0.5)
    │      - top-K selection (e.g., 200 masks by score)
    │      - resize to original image size
    │       │
    │       ▼
    │    mask_pred  [M, H, W]   (final mask proposals)
    │
    └──→ CLIP Adapter (MaskFormerClipAdapter, 既存コードを流用)
    │       │
    │       for each mask:
    │         1. crop masked region from image
    │         2. resize to 224x224
    │         3. fill non-mask region with mean pixel
    │         4. CLIP image encoder → image feature
    │         5. cosine similarity with text_features
    │       │
    │       ▼
    │    clip_cls  [M, num_classes]
    │
    ▼
    semseg = einsum("mc,mhw→chw", clip_cls, mask_pred)
    argmax → 最終セグメンテーション
```

#### シンプルになった点

- backbone + pixel decoder + transformer decoder + mask embedding → cosine sim のパスを SAM3 に一本化
- fusion が不要（分類は CLIP のみ、λ=0 相当）
- SAM3 は frozen pretrained model として使うので学習不要

## SAM3 のマスク生成戦略

SAM3 は Promptable Concept Segmentation (PCS) 用に設計されており、テキストプロンプトごとに物体を検出・セグメンテーションする。
クラス非依存のマスクプロポーザルとして使うには以下の戦略をとる。

### モード1: データセットクラス prompting (評価用)

データセットの全クラス名をテキストプロンプトとして SAM3 に与え、全クラスをカバーするマスクを生成する。

| データセット   | クラス数 | SAM3 prompts    |
| -------------- | -------- | --------------- |
| ADE20K-150     | 150      | 150 class names |
| COCO-Stuff-171 | 171      | 171 class names |

SAM3 は各クラスに対して複数のインスタンスマスクを返す（DETR detector、デフォルト query数 ~300）。
全クラスをバッチ処理することで効率的に推論できる。

### モード2: 大規模語彙 prompting (定性デモ用)

LVIS (1200カテゴリ) や ImageNet クラスなど、大規模な語彙でプロンプトすることで、未知のクラスに対しても高カバレッジなマスクを生成する。
CLIP による open-vocabulary 分類と組み合わせることで真の open-vocabulary segmentation を実現。

## ファイル構成

実装は `ov-seg` リポジトリ内で行う。以下は ov-seg のルートからの相対パス。

### 新規ファイル

| ファイル                                  | 内容                        |
| ----------------------------------------- | --------------------------- |
| `open_vocab_seg/modeling/sam3_adapter.py` | SAM3 モデルのラッパークラス |
| `open_vocab_seg/ovsam_seg_model.py`       | OVSAMSeg モデルクラス       |

### 修正ファイル

| ファイル                              | 修正内容                                  |
| ------------------------------------- | ----------------------------------------- |
| `open_vocab_seg/config.py`            | SAM3 用 config (`MODEL.SAM3`) 追加        |
| `open_vocab_seg/__init__.py`          | `OVSAMSeg` を `META_ARCH_REGISTRY` に登録 |
| `open_vocab_seg/utils/predictor.py`   | OVSAMSeg の predictor 追加                |
| `demo.py`                             | SAM3 config 読み込みの調整                |
| `train_net.py`                        | eval-only 時のモデル選択                  |
| `pyproject.toml` / `requirements.txt` | `transformers`, `accelerate` 追加         |

### 変更しないもの

- `open_vocab_seg/modeling/clip_adapter/` — MaskFormerClipAdapter をそのまま流用
- `open_vocab_seg/evaluation/generalized_sem_seg_evaluation.py` — mIoU 計算はモデル非依存
- `open_vocab_seg/data/` — データセット読み込みはそのまま
- `configs/*.yaml` — 新しい config を追加

## sam3_adapter.py 設計

```python
class SAM3ProposalGenerator(nn.Module):
    """
    SAM3 をマスクプロポーザル生成器としてラップする。
    Frozen inference-only。テキストプロンプトを受け取りマスクを返す。
    """

    def __init__(
        self,
        model_name: str = "facebook/sam3",
        device: str = "cuda",
        num_queries: int = 300,
        mask_threshold: float = 0.5,
        max_masks: int = 200,
    ):
        # SAM3 model と processor をロード
        self.model = Sam3Model.from_pretrained(model_name).to(device).eval()
        self.processor = Sam3Processor.from_pretrained(model_name)

    def forward(
        self,
        image: torch.Tensor,       # [3, H, W]  raw pixel (0-255)
        text_prompts: List[str],   # class names
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: raw input image (C, H, W), uint8 or float32
            text_prompts: list of text prompts for SAM3
        Returns:
            masks: [N, H, W]  binary masks
            scores: [N]  confidence scores
        """
        # 1. Processor で入力整形
        inputs = self.processor(
            images=image,
            text=text_prompts,
            return_tensors="pt"
        ).to(self.device)

        # 2. SAM3 forward
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 3. Post-process: masks → binary, filter by threshold
        results = self.processor.post_process_instance_segmentation(
            outputs, threshold=self.mask_threshold
        )[0]

        # 4. Select top-K masks by score
        masks = results["masks"]    # [N, H, W]  bool
        scores = results["scores"]  # [N]
        # top-K selection...

        return masks, scores
```

## ovsam_seg_model.py 設計

```python
@META_ARCH_REGISTRY.register()
class OVSAMSeg(nn.Module):
    """
    SAM3 をマスクプロポーザルに使った Open-Vocabulary Segmentation モデル。
    """

    def __init__(self, cfg):
        super().__init__()

        # SAM3 mask proposal generator
        self.sam3 = SAM3ProposalGenerator(
            model_name=cfg.MODEL.SAM3.MODEL_NAME,
            mask_threshold=cfg.MODEL.SAM3.MASK_THRESHOLD,
            max_masks=cfg.MODEL.SAM3.MAX_MASKS,
        )

        # CLIP adapter (OV-Seg から流用)
        text_templates = build_text_prompt(cfg.MODEL.CLIP_ADAPTER)
        self.clip_adapter = MaskFormerClipAdapter(
            cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME,
            text_templates,
            mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
            mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
            mask_thr=cfg.MODEL.CLIP_ADAPTER.MASK_THR,
            mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
            region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
            mask_prompt_depth=cfg.MODEL.CLIP_ADAPTER.MASK_PROMPT_DEPTH,
            mask_prompt_fwd=cfg.MODEL.CLIP_ADAPTER.MASK_PROMPT_FWD,
        )

    def forward(self, batched_inputs):
        # 1. Extract image and class names from input
        image = batched_inputs[0]["image"]  # [3, H, W]
        class_names = self._get_class_names(batched_inputs)

        # 2. SAM3: generate mask proposals
        sam3_masks, sam3_scores = self.sam3(image, class_names)

        if sam3_masks.shape[0] == 0:
            # No masks generated → return empty segmentation
            h, w = image.shape[-2:]
            return {"sem_seg": torch.zeros(len(class_names), h, w)}

        # 3. CLIP: classify each masked region
        clip_cls, _, valid_flag = self.clip_adapter(
            image, class_names, sam3_masks.float(), normalize=True
        )

        if clip_cls is None or clip_cls.shape[0] == 0:
            return {"sem_seg": torch.zeros(len(class_names), h, w)}

        # 4. Softmax
        clip_cls = F.softmax(clip_cls[:, :-1], dim=-1)

        # 5. Einsum fusion
        valid_masks = sam3_masks[valid_flag].float()
        semseg = torch.einsum("mc,mhw->chw", clip_cls, valid_masks)

        return {"sem_seg": semseg}
```

## config.py 追加項目

```python
def add_ovseg_config(cfg):
    # ... existing config ...

    # SAM3
    cfg.MODEL.SAM3 = CN()
    cfg.MODEL.SAM3.ENABLED = False          # True で OVSAMSeg に切り替え
    cfg.MODEL.SAM3.MODEL_NAME = "facebook/sam3"
    cfg.MODEL.SAM3.MASK_THRESHOLD = 0.5
    cfg.MODEL.SAM3.MAX_MASKS = 200
    cfg.MODEL.SAM3.PROMPT_SOURCE = "dataset"  # "dataset" or "lvis" or "custom"
```

## 実装手順

### Step 0: 環境準備

```bash
# ov-seg リポジトリで
pip install transformers accelerate
huggingface-cli login  # → token 入力 (SAM3 のアクセス権が必要)
```

### Step 1: sam3_adapter.py

- `SAM3ProposalGenerator` クラス
- モデルロード、forward、post-process の実装
- 単体テスト用のスクリプトを別途用意（任意画像とテキストプロンプトでマスク生成を確認）

### Step 2: ovsam_seg_model.py

- `OVSAMSeg` クラス
- SAM3 → CLIPAdapter → einsum のパイプライン
- demo inference と eval inference の両対応

### Step 3: config, registry, predictor

- config に SAM3 の設定追加
- `__init__.py` で register
- predictor/demo で新モデルを選択可能に

### Step 4: 評価

```bash
# ADE20K-150 で評価
python train_net.py \
    --eval-only \
    --config-file configs/ovsam_seg_swinB_vitL.yaml \
    MODEL.SAM3.ENABLED True \
    MODEL.WEIGHTS /path/to/pretrained/ovseg.pth
```

## 階層的セグメンテーションへの拡張（フェーズ2）

SAM3 の PVS (Perception Vision Segmentation) tracker head は、1プロンプトあたり 3 つのマスクを出力する（whole / part / sub-part）。
これを利用して、クエリの粒度に応じて適切な階層のマスクを選択する機構を後から追加できる。

```
Query: "car"         → whole mask (車全体)
Query: "left wheel"  → part mask  (左車輪)
Query: "tire"        → sub-part   (タイヤ)
```

これは Phase 1 完了後の拡張として計画する。

## 参考

- OV-Seg: https://github.com/facebookresearch/ov-seg
- SAM3: https://github.com/facebookresearch/sam3
- SAM3 paper: https://arxiv.org/abs/2511.16719
