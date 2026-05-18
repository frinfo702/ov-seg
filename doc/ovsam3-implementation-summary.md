# OVSAM3 実装概要

## 目的

既存の OVSeg (MaskFormer + CLIP) のマスク生成部分を SAM3 に置き換え、SAM3の高品質マスクプロポーザルをCLIPで分類するパイプラインを構築する。

## 構成

```
入力画像 → SAM3 (200マスク候補) → CLIP ViT-L/14 (領域分類) → セグメンテーションマップ
```

- マスク生成: SAM3 decoder (200 query)
- 領域分類: CLIP ViT-L/14 (各マスク領域を224×224にリサイズ→ViT→テキスト特徴と内積)
- 分類器の重み: OVSeg の Mask Prompt Tuning 済み weights を流用
- SAM3 の重み: HuggingFace (facebook/sam3) から自動ダウンロード

## 加えた修正

| ファイル                                     | 修正内容                                                                                                           |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `open_vocab_seg/modeling/__init__.py`        | `ovsam_seg_model` の import を追加 (META_ARCH_REGISTRY登録のため)                                                  |
| `open_vocab_seg/config.py`                   | `MODEL.SAM3.*` のデフォルト設定を追加                                                                              |
| `open_vocab_seg/modeling/sam3_proposal.py`   | SAM3 import (namespace shadowing対策), addmm_act dtype 不一致の monkey-patch, CUDA tensor→numpy 変換のハンドリング |
| `open_vocab_seg/modeling/ovsam_seg_model.py` | `log_first_n` の引数順序修正                                                                                       |
| `train_net.py`                               | `parse_known_args` + 奇数長opts対策                                                                                |
| `ovsam3-demo.py`                             | 新規作成。`parse_known_args` 採用、出力パス表示                                                                    |

## 実行方法

### デモ (単一画像)

```bash
uv run ovsam3-demo.py \
  --config-file configs/ovsam_seg_swinB_vitL.yaml \
  --input <image> \
  --class-names <category1 category2 ...> \
  --output <output.png> \
  MODEL.WEIGHTS checkpoints/ovseg_swinbase_vitL14_ft_mpt.pth
```

### 評価 (ADE20K val)

```bash
uv run train_net.py --eval-only \
  --config-file configs/ovsam_seg_swinB_vitL.yaml \
  MODEL.WEIGHTS checkpoints/ovseg_swinbase_vitL14_ft_mpt.pth
```

## 結果

| method | backbone (特徴抽出器) | head (分類器) | 分類器のtraining dataset | A-150 |
|--------|----------------------|---------------|-------------------------|-------|
| OVSAM3 (ours) | SAM3's intrinsic encoder | fine-tuned CLIP | COCO-Stuff-171 (pseudo masks + captions) | 25.677 |
| OVSeg | Swin-B | fine-tuned CLIP | COCO-Stuff-171 (pseudo masks + captions) | 29.6 |
| OVSeg | R101-c | fine-tuned CLIP | COCO-Stuff-171 (pseudo masks + captions) | 24.8 |
| SAM3 | SAM3's intrinsic encoder | SAM3's intrinsic decoder | SA-Co | 39.0 |

**注意**: OVSAM3 の分類器は OVSeg 由来の fine-tuned CLIP 重みをそのまま流用しており、OVSAM3 用に追加 fine-tuning は行っていない。また SAM3 の A-150=39.0 は評価方法が異なる可能性がある（ref: SAM3 report）。
