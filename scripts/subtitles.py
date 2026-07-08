"""
Step 6: Generate word-level subtitles for each scene using Whisper (free, local).
Also builds a merged .ass subtitle file for the final video.

Input:  data/storyboard.json, output/audio/scene_XXX.mp3
Output: output/subtitles/scene_XXX.srt, output/subtitles/final.ass
"""

import json
import whisper
from utils import ROOT, log, load_config, data_dir

ASS_HEADER_TEMPLATE = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, Bold, Outline, Shadow, Alignment, MarginV
Style: Default,{font},{font_size},{color},{outline_color},1,3,0,2,60

[Events]
Format: Layer, Start, End, Style, Text
"""


def format_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:01d}:{m:02d}:{s:05.2f}"


def main():
    cfg = load_config()

    storyboard_path = data_dir() / "storyboard.json"
    if not storyboard_path.exists():
        raise FileNotFoundError("data/storyboard.json not found. Run storyboard.py first.")

    with open(storyboard_path, "r") as f:
        storyboard = json.load(f)

    (ROOT / "output" / "subtitles").mkdir(parents=True, exist_ok=True)

    log("Loading Whisper model (base)... this can take a moment on first run.")
    model = whisper.load_model("base")

    sub_cfg = cfg["video"]["subtitle_style"]
    width, height = cfg["video"]["resolution"]
    ass_lines = [ASS_HEADER_TEMPLATE.format(
        width=width, height=height,
        font=sub_cfg["font"].split("/")[-1].replace(".ttf", ""),
        font_size=sub_cfg["font_size"],
        color=sub_cfg["color"],
        outline_color=sub_cfg["outline_color"],
    )]

    for scene in storyboard["scenes"]:
        audio_path = ROOT / scene["audio_path"]
        if not audio_path.exists():
            log(f"Audio missing for scene {scene['scene_number']}, skipping subtitles for it.", "WARN")
            continue

        log(f"Transcribing scene {scene['scene_number']}...")
        result = model.transcribe(str(audio_path), language=cfg["channel"]["language"], word_timestamps=False)

        scene_offset = scene["start_time"]
        for seg in result["segments"]:
            start = scene_offset + seg["start"]
            end = scene_offset + seg["end"]
            text = seg["text"].strip().replace("\n", " ")
            ass_lines.append(
                f"Dialogue: 0,{format_ass_time(start)},{format_ass_time(end)},Default,{text}"
            )

    out_path = ROOT / "output" / "subtitles" / "final.ass"
    with open(out_path, "w") as f:
        f.write("\n".join(ass_lines))

    log(f"Subtitles written to {out_path}")


if __name__ == "__main__":
    main()
