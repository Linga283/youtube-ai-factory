"""
Step 4: Generate an image for every scene in the storyboard.
Input:  data/storyboard.json
Output: output/images/scene_XXX.png (one per scene)
"""

import json
import time
from utils import ROOT, log, load_config, call_ai_image, data_dir


def main():
    cfg = load_config()

    storyboard_path = data_dir() / "storyboard.json"
    if not storyboard_path.exists():
        raise FileNotFoundError("data/storyboard.json not found. Run storyboard.py first.")

    with open(storyboard_path, "r") as f:
        storyboard = json.load(f)

    provider = cfg["image_generation"]["provider"]
    (ROOT / "output" / "images").mkdir(parents=True, exist_ok=True)

    total = len(storyboard["scenes"])
    for i, scene in enumerate(storyboard["scenes"], start=1):
        out_path = ROOT / scene["image_path"]
        if out_path.exists():
            log(f"[{i}/{total}] Scene {scene['scene_number']} image already exists, skipping.")
            continue

        log(f"[{i}/{total}] Generating image for scene {scene['scene_number']} via {provider}...")
        try:
            call_ai_image(scene["image_prompt"], provider, cfg, str(out_path))
        except Exception as e:
            log(f"Image generation failed for scene {scene['scene_number']}: {e}", "ERROR")
            raise

        # Gentle pacing to respect free-tier rate limits
        time.sleep(2)

    log(f"All {total} scene images generated in output/images/.")


if __name__ == "__main__":
    main()
