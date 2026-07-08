"""
Step 7: Build the final video with FFmpeg.
- Ken Burns zoom/pan on each scene image
- Narration audio per scene
- Background music ducked under narration
- Burned-in subtitles (final.ass)
- Optional intro/outro/logo

Input:  data/storyboard.json, output/images/*, output/audio/*, output/subtitles/final.ass
Output: output/final.mp4
"""

import json
import subprocess
import shutil
from utils import ROOT, log, load_config, data_dir


def run(cmd: list, description: str):
    log(f"Running: {description}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"FFmpeg command failed: {description}", "ERROR")
        log(result.stderr[-3000:], "ERROR")
        raise RuntimeError(f"FFmpeg step failed: {description}")
    return result


def build_scene_clip(scene: dict, cfg: dict):
    """Render one scene: still image + Ken Burns motion + narration audio."""
    width, height = cfg["video"]["resolution"]
    fps = cfg["video"]["fps"]
    duration = scene["duration_seconds"]
    movement = scene.get("camera_movement", "static")
    image_path = ROOT / scene["image_path"]
    audio_path = ROOT / scene["audio_path"]
    out_path = ROOT / scene["scene_video_path"]

    zoom_enabled = cfg["video"].get("ken_burns_zoom", True)

    # zoompan filter: scale up first for smooth zoom, then apply zoompan
    total_frames = int(duration * fps)

    if not zoom_enabled or movement == "static":
        zoompan = f"scale={width}:{height},setsar=1"
    elif movement == "zoom_in":
        zoompan = (
            f"scale=8000:-1,zoompan=z='min(zoom+0.0015,1.5)':d={total_frames}:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height},setsar=1"
        )
    elif movement == "zoom_out":
        zoompan = (
            f"scale=8000:-1,zoompan=z='if(eq(on,1),1.5,max(zoom-0.0015,1.0))':d={total_frames}:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height},setsar=1"
        )
    elif movement == "pan_left":
        zoompan = (
            f"scale=8000:-1,zoompan=z=1.2:d={total_frames}:"
            f"x='iw-(iw/zoom)-(on/{total_frames})*(iw-(iw/zoom))':y='ih/2-(ih/zoom/2)':s={width}x{height},setsar=1"
        )
    elif movement == "pan_right":
        zoompan = (
            f"scale=8000:-1,zoompan=z=1.2:d={total_frames}:"
            f"x='(on/{total_frames})*(iw-(iw/zoom))':y='ih/2-(ih/zoom/2)':s={width}x{height},setsar=1"
        )
    else:
        zoompan = f"scale={width}:{height},setsar=1"

    has_audio = audio_path.exists()

    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", str(image_path)]
    if has_audio:
        cmd += ["-i", str(audio_path)]
    cmd += ["-vf", zoompan, "-r", str(fps), "-t", str(duration), "-pix_fmt", "yuv420p"]
    if has_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(out_path)]

    run(cmd, f"Build scene {scene['scene_number']} clip ({movement}, {duration}s)")


def concat_scenes(storyboard: dict, cfg: dict, out_path):
    """Concatenate all scene clips (and optional intro/outro) into one file."""
    list_file = ROOT / "output" / "concat_list.txt"
    branding = cfg["branding"]

    entries = []
    intro = ROOT / branding.get("intro_video", "")
    if branding.get("intro_video") and intro.exists():
        entries.append(intro)

    for scene in storyboard["scenes"]:
        entries.append(ROOT / scene["scene_video_path"])

    outro = ROOT / branding.get("outro_video", "")
    if branding.get("outro_video") and outro.exists():
        entries.append(outro)

    with open(list_file, "w") as f:
        for e in entries:
            f.write(f"file '{e.as_posix()}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-c", "copy", str(out_path),
    ]
    run(cmd, "Concatenate all scene clips")


def add_music_and_subtitles(video_in, video_out, cfg: dict):
    branding = cfg["branding"]
    music_path = ROOT / branding.get("background_music", "")
    subs_path = ROOT / "output" / "subtitles" / "final.ass"
    music_db = branding.get("music_volume_db", -22)

    has_music = branding.get("background_music") and music_path.exists()
    has_subs = subs_path.exists()

    cmd = ["ffmpeg", "-y", "-i", str(video_in)]
    if has_music:
        cmd += ["-stream_loop", "-1", "-i", str(music_path)]

    filter_parts = []
    video_label = "0:v"
    if has_subs:
        # ass filter needs a path relative-safe string
        filter_parts.append(f"[0:v]ass={subs_path.as_posix()}[vout]")
        video_label = "vout"

    audio_label = "0:a"
    if has_music:
        filter_parts.append(f"[1:a]volume={music_db}dB[music]")
        filter_parts.append(f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]")
        audio_label = "aout"

    if filter_parts:
        cmd += ["-filter_complex", ";".join(filter_parts)]
        cmd += ["-map", f"[{video_label}]" if has_subs else "0:v"]
        cmd += ["-map", f"[{audio_label}]" if has_music else "0:a"]
    cmd += ["-c:v", "libx264", "-c:a", "aac", "-shortest", str(video_out)]

    run(cmd, "Add background music + burned-in subtitles")


def main():
    cfg = load_config()

    storyboard_path = data_dir() / "storyboard.json"
    if not storyboard_path.exists():
        raise FileNotFoundError("data/storyboard.json not found. Run storyboard.py first.")
    with open(storyboard_path, "r") as f:
        storyboard = json.load(f)

    output_dir = ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(storyboard["scenes"])
    for i, scene in enumerate(storyboard["scenes"], start=1):
        log(f"[{i}/{total}] Building scene {scene['scene_number']} clip...")
        build_scene_clip(scene, cfg)

    concat_path = output_dir / "concatenated.mp4"
    concat_scenes(storyboard, cfg, concat_path)

    final_path = output_dir / "final.mp4"
    add_music_and_subtitles(concat_path, final_path, cfg)

    log(f"Final video ready: {final_path}")


if __name__ == "__main__":
    main()
