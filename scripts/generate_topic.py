"""
Step 1: Generate a fresh video topic + title, avoiding recent repeats.
Output: data/topic.json
"""

from utils import (
    ROOT, log, load_config, load_prompts, call_ai_text, extract_json,
    load_history, data_dir,
)
import json


def main():
    cfg = load_config()
    prompts = load_prompts()

    history = load_history(cfg)
    recent_topics = [h.get("topic", "") for h in history[-20:]]

    prompt = prompts["topic_generation"].format(
        channel_name=cfg["channel"]["name"],
        niche=cfg["channel"]["niche"],
        recent_topics=", ".join(recent_topics) if recent_topics else "none yet",
        duration=cfg["channel"]["target_duration_minutes"],
    )

    provider = cfg["script_generation"]["provider"]
    model = cfg["script_generation"]["gemini_model"] if provider == "gemini" else cfg["script_generation"]["openai_model"]

    log(f"Requesting topic from {provider} ({model})...")
    raw = call_ai_text(prompt, provider, model, max_tokens=500)
    topic_data = extract_json(raw)

    for key in ("topic", "title", "learning_goal"):
        if key not in topic_data:
            raise ValueError(f"Missing '{key}' in topic generation output: {topic_data}")

    out_path = data_dir() / "topic.json"
    with open(out_path, "w") as f:
        json.dump(topic_data, f, indent=2)

    log(f"Topic generated: {topic_data['title']}")
    log(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
