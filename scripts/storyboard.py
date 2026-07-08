"""
Step 3: Turn the raw script into a finalized storyboard:
- appends consistent style suffix to every image prompt
- assigns file paths for each scene's audio/image/subtitle output
- computes running timeline (start/end times) used later by make_video.py

Input:  data/script.json
Output: data/storyboard.json
"""

import json
from utils import log, load_config, load_prompts, data_dir


def main():
    cfg = load_config()
    prompts = load_prompts()

    script_path = data_dir() / "script.json"
    if not script_path.exists():
        raise FileNotFoundError("data/script.json not found. Run generate_script.py first.")

    with open(script_path, "r") as f:
        script_data = json.load(f)

    style_suffix = cfg["image_generation"]["style_suffix"]
    running_time = 0.0
    storyboard_scenes = []

    for scene in script_data["scenes"]:
        n = scene["scene_number"]
        final_image_prompt = f"{scene['image_prompt']}, {style_suffix}"
        duration = float(scene["duration_seconds"])

        storyboard_scenes.append({
            "scene_number": n,
            "narration": scene["narration"],
            "image_prompt": final_image_prompt,
            "camera_movement": scene.get("camera_movement", "static"),
            "sound_effect": scene.get("sound_effect"),
            "duration_seconds": duration,
            "start_time": round(running_time, 2),
            "end_time": round(running_time + duration, 2),
            "image_path": f"output/images/scene_{n:03d}.png",
            "audio_path": f"output/audio/scene_{n:03d}.mp3",
            "scene_video_path": f"output/scene_{n:03d}.mp4",
        })
        running_time += duration

    storyboard = {
        "title": script_data["title"],
        "topic": script_data.get("topic"),
        "learning_goal": script_data.get("learning_goal"),
        "total_duration_seconds": round(running_time, 2),
        "scenes": storyboard_scenes,
    }

    out_path = data_dir() / "storyboard.json"
    with open(out_path, "w") as f:
        json.dump(storyboard, f, indent=2)

    log(f"Storyboard finalized: {len(storyboard_scenes)} scenes, "
        f"~{running_time/60:.1f} min total.")
    log(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
