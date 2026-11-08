# Feedback system

A local Streamlit app for collecting three 1–5 ratings, optional written
feedback, and optional spoken feedback transcribed with Faster-Whisper. Spoken
feedback also gets a local topic and sentiment overview.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

The first voice transcription loads the `base` Faster-Whisper model and may
download it. Transcription runs on the app server using CPU inference. Ratings
and comments are kept in the current Streamlit session; they are not saved to a
database. The transcript is written to a temporary server-side text file for
download and is removed when you clear the session. Temporary files may also be
removed by the operating system.

On the first voice submission, KeyBERT may download the
`all-MiniLM-L6-v2` embedding model. It extracts up to five topics locally;
VADER's local sentiment lexicon assigns positive, negative, neutral, or mixed
sentiment to the matching spoken clauses. Each topic's description is quoted
from the transcript rather than generated, so the overview stays grounded in
the speaker's words. Audio and transcript analysis are not sent to a cloud AI
service.

If you already installed the app before changing dependencies, update the
environment with `python -m pip install -r requirements.txt` before restarting
Streamlit. The PyAV version is capped below 19 to remain compatible with
Faster-Whisper releases that still pass the removed `metadata_errors` option.
