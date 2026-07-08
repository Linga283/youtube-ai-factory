"""
Step 5: Generate narration audio for every scene using Microsoft Edge TTS (free).
Input:  data/storyboard.json
Output: output/audio/scene_XXX.mp3
"""

import asyncio
import json
from utils import ROOT, log, load_config, data_dir

import edge_tts


async def synthesize(text: str, voice: str, rate: str, pitch: str, out_path: str):
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(out_path)


async def main_async():
    cfg = load_config()

    storyboard_path = data_dir() / "storyboard.json"
    if not storyboard_path.exists():
        raise FileNotFoundError("data/storyboard.json not found. Run storyboard.py first.")

    with open(storyboard_path, "r") as f:
        storyboard = json.load(f)

    (ROOT / "output" / "audio").mkdir(parents=True, exist_ok=True)

    voice = cfg["narrator"]["voice_name"]
    rate = cfg["narrator"]["rate"]
    pitch = cfg["narrator"]["pitch"]

    total = len(storyboard["scenes"])
    for i, scene in enumerate(storyboard["scenes"], start=1):
        out_path = ROOT / scene["audio_path"]
        if out_path.exists():
            log(f"[{i}/{total}] Scene {scene['scene_number']} audio already exists, skipping.")
            continue

        log(f"[{i}/{total}] Synthesizing narration for scene {scene['scene_number']}...")
        await synthesize(scene["narration"], voice, rate, pitch, str(out_path))

    log(f"All {total} narration clips generated in output/audio/.")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
