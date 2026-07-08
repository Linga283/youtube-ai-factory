"""
Step 9: Generate SEO metadata (title, description, tags, hashtags).
Input:  data/storyboard.json
Output: data/seo.json
"""

import json
from utils import log, load_config, load_prompts, call_ai_text, extract_json, data_dir


def main():
    cfg = load_config()
    prompts = load_prompts()

    storyboard_path = data_dir() / "storyboard.json"
    if not storyboard_path.exists():
        raise FileNotFoundError("data/storyboard.json not found. Run storyboard.py first.")

    with open(storyboard_path, "r") as f:
        storyboard = json.load(f)

    full_narration = " ".join(s["narration"] for s in storyboard["scenes"])

    prompt = prompts["seo_generation"].format(
        title=storyboard["title"],
        learning_goal=storyboard.get("learning_goal", ""),
        full_narration=full_narration[:3000],  # keep prompt reasonably sized
    )

    provider = cfg["script_generation"]["provider"]
    model = cfg["script_generation"]["gemini_model"] if provider == "gemini" else cfg["script_generation"]["openai_model"]

    log(f"Requesting SEO metadata from {provider} ({model})...")
    raw = call_ai_text(prompt, provider, model, max_tokens=1000)
    seo_data = extract_json(raw)

    for key in ("title", "description", "tags"):
        if key not in seo_data:
            raise ValueError(f"Missing '{key}' in SEO output: {seo_data}")

    out_path = data_dir() / "seo.json"
    with open(out_path, "w") as f:
        json.dump(seo_data, f, indent=2)

    log(f"SEO metadata generated: {seo_data['title']}")
    log(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
