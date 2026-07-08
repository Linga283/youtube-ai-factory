"""
Step 2: Generate the full scene-by-scene script from the chosen topic.
Input:  data/topic.json
Output: data/script.json
"""

import json
from utils import (
    log, load_config, load_prompts, call_ai_text, extract_json, data_dir,
)


def main():
    cfg = load_config()
    prompts = load_prompts()

    topic_path = data_dir() / "topic.json"
    if not topic_path.exists():
        raise FileNotFoundError("data/topic.json not found. Run generate_topic.py first.")

    with open(topic_path, "r") as f:
        topic_data = json.load(f)

    prompt = prompts["script_generation"].format(
        persona=cfg["narrator"]["persona"],
        title=topic_data["title"],
        learning_goal=topic_data["learning_goal"],
        duration=cfg["channel"]["target_duration_minutes"],
        max_scenes=cfg["script_generation"]["max_scenes"],
    )

    provider = cfg["script_generation"]["provider"]
    model = cfg["script_generation"]["gemini_model"] if provider == "gemini" else cfg["script_generation"]["openai_model"]

    log(f"Requesting script from {provider} ({model})...")
    raw = call_ai_text(prompt, provider, model, max_tokens=4096)
    script_data = extract_json(raw)

    if "scenes" not in script_data or not isinstance(script_data["scenes"], list):
        raise ValueError(f"Script output missing 'scenes' list: {script_data}")

    # Basic validation / normalization
    for i, scene in enumerate(script_data["scenes"], start=1):
        scene.setdefault("scene_number", i)
        scene.setdefault("sound_effect", None)
        scene.setdefault("camera_movement", "static")
        scene["duration_seconds"] = max(6, min(12, int(scene.get("duration_seconds", 8))))

    out_path = data_dir() / "script.json"
    with open(out_path, "w") as f:
        json.dump({**topic_data, **script_data}, f, indent=2)

    log(f"Script generated with {len(script_data['scenes'])} scenes.")
    log(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
