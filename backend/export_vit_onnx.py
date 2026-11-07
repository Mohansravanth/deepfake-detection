"""Export the dima806 ViT deepfake classifier to ONNX for fast CPU inference."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "models" / "vit_deepfake.onnx"
REPO_ID = "dima806/deepfake_vs_real_image_detection"


def main():
    import torch
    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(REPO_ID).eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        (dummy,),
        str(OUT),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print("Saved", OUT, OUT.stat().st_size / 1e6, "MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
