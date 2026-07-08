"""
Step 8: Generate an eye-catching thumbnail for the video.
Input:  data/storyboard.json
Output: output/thumbnail.jpg
"""

import json
from PIL import Image, ImageDraw, ImageFont
from utils import ROOT, log, load_config, load_prompts, call_ai_image, data_dir


def main():
    cfg = load_config()
    prompts = load_prompts()

    storyboard_path = data_dir() / "storyboard.json"
    if not storyboard_path.exists():
        raise FileNotFoundError("data/storyboard.json not found. Run storyboard.py first.")

    with open(storyboard_path, "r") as f:
        storyboard = json.load(f)

    title = storyboard["title"]
    prompt = prompts["thumbnail_prompt"].format(title=title)
    provider = cfg["image_generation"]["provider"]

    raw_path = ROOT / "output" / "thumbnail_raw.png"
    log(f"Generating thumbnail base image via {provider}...")
    call_ai_image(prompt, provider, cfg, str(raw_path))

    # Resize to standard YouTube thumbnail size and overlay bold title text
    img = Image.open(raw_path).convert("RGB")
    img = img.resize((1280, 720))
    draw = ImageDraw.Draw(img)

    font_path = ROOT / cfg["video"]["subtitle_style"]["font"]
    try:
        font = ImageFont.truetype(str(font_path), 80) if font_path.exists() else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Simple bold outlined text near the bottom
    text = title.upper()
    x, y = 40, 560
    outline_range = 4
    for dx in range(-outline_range, outline_range + 1, 2):
        for dy in range(-outline_range, outline_range + 1, 2):
            draw.text((x + dx, y + dy), text, font=font, fill="black")
    draw.text((x, y), text, font=font, fill="yellow")

    out_path = ROOT / "output" / "thumbnail.jpg"
    img.save(out_path, quality=95)
    log(f"Thumbnail saved to {out_path}")


if __name__ == "__main__":
    main()
