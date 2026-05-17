"""
SAM3 subprocess worker.
Called by SAM3ProposalGenerator via subprocess when sam3 package
cannot be imported in the main Python environment.
"""

import argparse
import json

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=str, required=True)
    parser.add_argument("--output-npz", type=str, required=True)
    args = parser.parse_args()

    with open(args.input_json) as f:
        params: dict = json.load(f)

    image_path: str = params["image_path"]
    text_prompts: list[str] = params["text_prompts"]
    mask_threshold: float = params.get("mask_threshold", 0.5)
    max_masks: int = params.get("max_masks", 200)

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    model = model.eval()

    image = Image.open(image_path).convert("RGB")
    inference_state = processor.set_image(image)

    all_masks: list[np.ndarray] = []
    all_scores: list[float] = []

    for prompt in text_prompts:
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        masks: list[np.ndarray] = output["masks"]
        scores: list[float] = output["scores"]
        for m, s in zip(masks, scores):
            if float(s) >= mask_threshold:
                all_masks.append(np.asarray(m, dtype=np.float32))
                all_scores.append(float(s))

    if len(all_scores) > max_masks:
        idx = np.argsort(all_scores)[-max_masks:]
        all_masks = [all_masks[i] for i in idx]
        all_scores = [all_scores[i] for i in idx]

    if len(all_masks) == 0:
        np.savez_compressed(
            args.output_npz,
            masks=np.empty((0, 0, 0), dtype=np.float32),
            scores=np.empty(0, dtype=np.float32),
        )
    else:
        np.savez_compressed(
            args.output_npz,
            masks=np.stack(all_masks, axis=0),
            scores=np.array(all_scores, dtype=np.float32),
        )

    result = {"num_masks": len(all_masks)}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
