"""
Step 10: Upload the final video to YouTube via the Data API v3.
Input:  output/final.mp4, output/thumbnail.jpg, data/seo.json, data/storyboard.json
Output: uploaded video (also appends an entry to history)

Auth model:
- Uses a refresh token stored in the YOUTUBE_REFRESH_TOKEN secret, plus
  YOUTUBE_CLIENT_ID / YOUTUBE_CLIENT_SECRET, to mint an access token at
  runtime. This avoids any interactive browser login inside CI.
  See README.md for the one-time local setup to obtain the refresh token.
"""

import json
import os
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from utils import ROOT, log, load_config, get_env, data_dir, save_history_entry


def get_authenticated_service():
    creds = Credentials(
        token=None,
        refresh_token=get_env("YOUTUBE_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=get_env("YOUTUBE_CLIENT_ID"),
        client_secret=get_env("YOUTUBE_CLIENT_SECRET"),
        scopes=["https://www.googleapis.com/auth/youtube.upload",
                "https://www.googleapis.com/auth/youtube"],
    )
    creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)


def get_or_create_playlist(youtube, playlist_name: str) -> str:
    request = youtube.playlists().list(part="snippet", mine=True, maxResults=50)
    response = request.execute()
    for item in response.get("items", []):
        if item["snippet"]["title"] == playlist_name:
            return item["id"]

    log(f"Playlist '{playlist_name}' not found, creating it...")
    create_request = youtube.playlists().insert(
        part="snippet,status",
        body={
            "snippet": {"title": playlist_name, "description": f"Auto-generated playlist: {playlist_name}"},
            "status": {"privacyStatus": "public"},
        },
    )
    result = create_request.execute()
    return result["id"]


def main():
    cfg = load_config()
    upload_cfg = cfg["upload"]

    seo_path = data_dir() / "seo.json"
    storyboard_path = data_dir() / "storyboard.json"
    video_path = ROOT / "output" / "final.mp4"
    thumbnail_path = ROOT / "output" / "thumbnail.jpg"

    for p in (seo_path, storyboard_path, video_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}. Run earlier pipeline steps first.")

    with open(seo_path, "r") as f:
        seo = json.load(f)
    with open(storyboard_path, "r") as f:
        storyboard = json.load(f)

    youtube = get_authenticated_service()

    body = {
        "snippet": {
            "title": seo["title"][:100],
            "description": seo["description"] + "\n\n" + " ".join(seo.get("hashtags", [])),
            "tags": seo.get("tags", [])[:15],
            "categoryId": upload_cfg["category_id"],
        },
        "status": {
            "privacyStatus": upload_cfg["privacy_status"],
            "selfDeclaredMadeForKids": cfg["channel"]["made_for_kids"],
        },
    }

    log(f"Uploading video: {body['snippet']['title']}")
    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            log(f"Upload progress: {int(status.progress() * 100)}%")

    video_id = response["id"]
    log(f"Upload complete. Video ID: {video_id}")

    if thumbnail_path.exists():
        log("Setting custom thumbnail...")
        youtube.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(str(thumbnail_path))).execute()

    playlist_name = upload_cfg.get("default_playlist")
    if playlist_name:
        playlist_id = get_or_create_playlist(youtube, playlist_name)
        youtube.playlistItems().insert(
            part="snippet",
            body={"snippet": {"playlistId": playlist_id,
                               "resourceId": {"kind": "youtube#video", "videoId": video_id}}},
        ).execute()
        log(f"Added to playlist '{playlist_name}'.")

    save_history_entry(cfg, {
        "topic": storyboard.get("topic"),
        "title": seo["title"],
        "video_id": video_id,
        "url": f"https://youtu.be/{video_id}",
    })

    log(f"Published: https://youtu.be/{video_id}")


if __name__ == "__main__":
    main()
