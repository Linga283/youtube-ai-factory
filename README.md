# YouTube AI Factory

Fully automated pipeline: topic → script → images → narration → subtitles →
video → thumbnail → SEO → YouTube upload. Runs on GitHub Actions, free-tier
friendly.

This README is written for **you** — everything the code needs is already
built. You only need to do the setup steps below once.

---

## What's already done (nothing for you to build)

- All pipeline scripts (`scripts/*.py`)
- Config files (`config/channel.yml`, `config/prompts.yml`) — this is where
  you tweak niche, persona, voice, image style, etc. **without touching code**
- GitHub Actions workflow (`.github/workflows/youtube.yml`)
- FFmpeg video builder tested locally (Ken Burns zoom, concat, music ducking,
  burned-in subtitles) — confirmed working on synthetic test assets
- Thumbnail text-overlay logic tested locally
- JSON parsing / provider-switch logic for Gemini ↔ OpenAI

## What I could NOT test from my side (needs your environment)

- **Edge TTS narration** — the sandbox I built this in blocks the network
  domain Edge TTS needs (`speech.platform.bing.com`). It will work fine on
  GitHub Actions (open network), but I couldn't prove it live. First run,
  watch this step's log.
- **Gemini / OpenAI image generation** — needs a real API key, which you said
  you'd add later.
- **YouTube upload** — needs your OAuth credentials, can only be tested from
  your own Google account.

---

## Setup steps (do these once)

### 1. Push this repo to GitHub
Create a new repo (public or private, doesn't matter) and push this folder
to it.

### 2. Get a Gemini API key (free tier)
1. Go to https://aistudio.google.com/apikey
2. Create an API key
3. In your GitHub repo: **Settings → Secrets and variables → Actions → New
   repository secret**
4. Name: `GEMINI_API_KEY`, value: your key

*(If you'd rather use OpenAI instead, get a key from
https://platform.openai.com/api-keys, add it as secret `OPENAI_API_KEY`, and
change `provider: "gemini"` to `provider: "openai"` in
`config/channel.yml` under both `image_generation` and `script_generation`.)*

### 3. Set up YouTube upload credentials (one-time, ~10 minutes)
This is the fiddly part — YouTube requires OAuth, not a simple API key.

1. Go to https://console.cloud.google.com/ → create a new project
2. Enable the **YouTube Data API v3** for that project
3. Go to **APIs & Services → Credentials → Create Credentials → OAuth client
   ID**. Application type: **Desktop app**. Download the JSON — note the
   `client_id` and `client_secret`.
4. On your own computer (not GitHub), run this one-time script to get a
   refresh token:

   ```bash
   pip install google-auth-oauthlib
   python - <<'EOF'
   from google_auth_oauthlib.flow import InstalledAppFlow

   SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
             "https://www.googleapis.com/auth/youtube"]

   flow = InstalledAppFlow.from_client_config(
       {
           "installed": {
               "client_id": "YOUR_CLIENT_ID",
               "client_secret": "YOUR_CLIENT_SECRET",
               "auth_uri": "https://accounts.google.com/o/oauth2/auth",
               "token_uri": "https://oauth2.googleapis.com/token",
               "redirect_uris": ["http://localhost"],
           }
       },
       SCOPES,
   )
   creds = flow.run_local_server(port=0)
   print("REFRESH TOKEN:", creds.refresh_token)
   EOF
   ```

   This opens a browser, asks you to log into the YouTube channel's Google
   account, and prints a refresh token in your terminal.

5. Add three more GitHub secrets:
   - `YOUTUBE_CLIENT_ID`
   - `YOUTUBE_CLIENT_SECRET`
   - `YOUTUBE_REFRESH_TOKEN`

### 4. (Optional) Add branding assets
- Drop an intro clip at `templates/intro.mp4`, outro at `templates/outro.mp4`
- Drop background music at `assets/music/kids_bgm.mp3`
- Drop a logo at `assets/logos/logo.png`
- Drop a real font at `assets/fonts/Poppins-Bold.ttf` (subtitles/thumbnail
  fall back to a plain default font if missing, but it looks much better
  with a real one — e.g. download free from Google Fonts)

All of these are optional — the pipeline skips anything that isn't present.

### 5. Run it
- Go to your repo → **Actions** tab → **YouTube AI Factory** → **Run
  workflow**
- First time, tick "skip_upload: true" so you can check the generated video
  as a downloadable artifact before it ever touches your real channel
- Once you're happy, run it again with skip_upload: false to actually publish

It's also scheduled to run daily at 09:00 UTC (edit the `cron:` line in
`.github/workflows/youtube.yml` or delete it if you only want manual runs).

---

## Customizing for a different niche

Everything is driven by `config/channel.yml` and `config/prompts.yml` — you
should not need to touch any `.py` file to:
- Change niche (kids → facts → motivation → coding, etc.)
- Change narrator voice/persona
- Change video length or number of scenes
- Switch between Gemini and OpenAI for text or images
- Change privacy status (public/unlisted/private) or scheduling

To run a second, different channel, copy this whole repo folder, edit its
`config/channel.yml`, and set up a separate set of GitHub secrets pointing at
a different YouTube channel.

## Running one step at a time (debugging)

```bash
cd scripts
python main.py --only topic       # just generate a topic
python main.py --only script      # just generate the script
python main.py --only images      # just generate images
python main.py --skip-upload      # run everything except the YouTube upload
```

## Repo structure

```
youtube-ai-factory/
├── config/
│   ├── channel.yml       ← main knobs: niche, voice, image provider, etc.
│   └── prompts.yml       ← all AI prompt templates
├── assets/               ← your music/fonts/logo (optional)
├── scripts/              ← all pipeline Python code
├── templates/            ← your intro/outro clips (optional)
├── output/               ← generated per-run (images, audio, final.mp4...)
├── data/                 ← generated per-run (topic.json, script.json...)
│   └── video_history.json  ← persists across runs, avoids topic repeats
└── .github/workflows/youtube.yml  ← the automation itself
```
