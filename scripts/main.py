"""
Main orchestrator: runs the full pipeline end-to-end, step by step.
Usage:
    python scripts/main.py                # run everything
    python scripts/main.py --skip-upload   # generate video but don't publish
    python scripts/main.py --only images   # run just one step (for debugging)
"""

import argparse
import sys
import time
from utils import log

STEPS = [
    ("topic", "generate_topic"),
    ("script", "generate_script"),
    ("storyboard", "storyboard"),
    ("images", "generate_images"),
    ("voice", "generate_voice"),
    ("subtitles", "subtitles"),
    ("video", "make_video"),
    ("thumbnail", "thumbnail"),
    ("seo", "generate_seo"),
    ("upload", "upload"),
]


def run_step(module_name: str):
    module = __import__(module_name)
    module.main()


def main():
    parser = argparse.ArgumentParser(description="YouTube AI Factory pipeline runner")
    parser.add_argument("--skip-upload", action="store_true", help="Build the video but skip YouTube upload")
    parser.add_argument("--only", type=str, default=None,
                         help="Run only one named step: " + ", ".join(s[0] for s in STEPS))
    args = parser.parse_args()

    steps_to_run = STEPS
    if args.only:
        steps_to_run = [s for s in STEPS if s[0] == args.only]
        if not steps_to_run:
            log(f"Unknown step '{args.only}'. Valid steps: {[s[0] for s in STEPS]}", "ERROR")
            sys.exit(1)
    elif args.skip_upload:
        steps_to_run = [s for s in STEPS if s[0] != "upload"]

    start = time.time()
    for name, module_name in steps_to_run:
        log(f"===== STEP: {name} =====")
        try:
            run_step(module_name)
        except Exception as e:
            log(f"Pipeline failed at step '{name}': {e}", "ERROR")
            sys.exit(1)

    elapsed = time.time() - start
    log(f"Pipeline complete in {elapsed/60:.1f} minutes.")


if __name__ == "__main__":
    main()
