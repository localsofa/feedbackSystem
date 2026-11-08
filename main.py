from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from faster_whisper import WhisperModel
from keybert import KeyBERT
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


MODEL_SIZE = "base"
TRANSCRIPT_KEY = "transcript_path"
SUBMISSION_KEY = "latest_submission"
TOPIC_MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_resource(show_spinner="Loading speech recognition model...")
def load_whisper_model() -> WhisperModel:
    return WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


@st.cache_resource(show_spinner="Loading local topic analysis model...")
def load_topic_model() -> KeyBERT:
    return KeyBERT(model=TOPIC_MODEL_NAME)


@st.cache_resource
def load_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def split_feedback_clauses(transcript: str) -> list[str]:
    clauses = re.split(
        r"(?<=[.!?;])\s+|[,;\n]+|\s+\b(?:but|however|although|though|yet)\b\s+",
        transcript,
        flags=re.IGNORECASE,
    )
    return [clause.strip() for clause in clauses if clause.strip()]


def normalize_topic_text(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def analyze_feedback_topics(transcript: str) -> list[dict[str, str]]:
    clauses = split_feedback_clauses(transcript)
    if not clauses:
        return []

    keywords = load_topic_model().extract_keywords(
        transcript,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        use_mmr=True,
        diversity=0.65,
        top_n=8,
    )
    sentiment_analyzer = load_sentiment_analyzer()
    topics: list[dict[str, str]] = []
    seen_topics: list[str] = []

    for keyword, _ in keywords:
        normalized_keyword = normalize_topic_text(keyword)
        if not normalized_keyword or any(
            normalized_keyword == topic
            or normalized_keyword in topic.split()
            or topic in normalized_keyword.split()
            for topic in seen_topics
        ):
            continue

        matching_clauses = [
            clause
            for clause in clauses
            if f" {normalized_keyword} " in f" {normalize_topic_text(clause)} "
        ]
        if not matching_clauses:
            continue

        sentiment_labels: set[str] = set()
        for clause in matching_clauses:
            score = sentiment_analyzer.polarity_scores(clause)["compound"]
            if score >= 0.05:
                sentiment_labels.add("Positive")
            elif score <= -0.05:
                sentiment_labels.add("Negative")
            else:
                sentiment_labels.add("Neutral")

        if "Positive" in sentiment_labels and "Negative" in sentiment_labels:
            sentiment = "Mixed"
        elif "Positive" in sentiment_labels:
            sentiment = "Positive"
        elif "Negative" in sentiment_labels:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        description = " ".join(f'"{clause}"' for clause in matching_clauses[:3])
        topics.append(
            {
                "Topic": keyword,
                "Sentiment": sentiment,
                "Description": description,
            }
        )
        seen_topics.append(normalized_keyword)
        if len(topics) == 5:
            break

    return topics


def remove_temporary_transcript() -> None:
    transcript_path = st.session_state.pop(TRANSCRIPT_KEY, None)
    if transcript_path:
        try:
            Path(transcript_path).unlink()
        except FileNotFoundError:
            pass


def transcribe_audio(audio_bytes: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as audio_file:
        audio_file.write(audio_bytes)
        audio_path = Path(audio_file.name)

    try:
        model = load_whisper_model()
        segments, _ = model.transcribe(str(audio_path))
        return " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        try:
            audio_path.unlink()
        except FileNotFoundError:
            pass


def save_temporary_transcript(transcript: str) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
        prefix="feedback-transcript-",
        delete=False,
    ) as transcript_file:
        transcript_file.write(transcript)
        return transcript_file.name


def main() -> None:
    st.set_page_config(page_title="Feedback", page_icon="💬")
    st.title("Share your feedback")
    st.write(
        "Rate your experience and, if you like, record a voice comment. "
        "Voice recordings are transcribed and analyzed on this server; "
        "recordings are not retained."
    )
    st.caption(
        "The first voice analysis downloads a local topic model. "
        "Audio and transcripts are not sent to a cloud AI service."
    )

    if st.button("Clear this session's feedback"):
        remove_temporary_transcript()
        st.session_state.pop(SUBMISSION_KEY, None)
        st.session_state.pop("spoken_feedback", None)

    audio = st.audio_input("Optional spoken feedback", key="spoken_feedback")

    with st.form("feedback_form"):
        overall_rating = st.slider("Overall experience", 1, 5, 3)
        ease_rating = st.slider("Ease of use", 1, 5, 3)
        quality_rating = st.slider("Quality", 1, 5, 3)
        written_feedback = st.text_area("Anything else you'd like to share?")
        submitted = st.form_submit_button("Submit feedback")

    if submitted:
        remove_temporary_transcript()
        transcript = ""
        topics: list[dict[str, str]] = []
        transcription_error: str | None = None
        analysis_error: str | None = None

        if audio is not None:
            suffix = Path(audio.name).suffix.lower()
            if suffix not in {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac"}:
                suffix = ".wav"
            try:
                transcript = transcribe_audio(audio.getvalue(), suffix)
            except (OSError, RuntimeError, ValueError) as error:
                transcription_error = str(error)

            if transcript:
                st.session_state[TRANSCRIPT_KEY] = save_temporary_transcript(
                    transcript
                )
                try:
                    topics = analyze_feedback_topics(transcript)
                except (
                    ImportError,
                    OSError,
                    RuntimeError,
                    ValueError,
                    requests.exceptions.RequestException,
                ) as error:
                    analysis_error = str(error)
            elif transcription_error is None:
                st.warning("No speech was detected in the recording.")

        st.session_state[SUBMISSION_KEY] = {
            "overall": overall_rating,
            "ease": ease_rating,
            "quality": quality_rating,
            "written_feedback": written_feedback.strip(),
            "transcript": transcript,
            "topics": topics,
            "analysis_error": analysis_error,
        }

        if transcription_error:
            st.error(f"Could not transcribe the recording: {transcription_error}")
        st.success("Your ratings and written feedback have been submitted.")

    submission: dict[str, Any] | None = st.session_state.get(SUBMISSION_KEY)
    if submission is not None:
        st.subheader("Your latest feedback")
        st.write(
            f"Overall: {submission['overall']}/5 · "
            f"Ease of use: {submission['ease']}/5 · "
            f"Quality: {submission['quality']}/5"
        )
        if submission["written_feedback"]:
            st.write(submission["written_feedback"])
        if submission["transcript"]:
            st.subheader("Spoken feedback overview")
            if submission["topics"]:
                st.dataframe(
                    submission["topics"],
                    hide_index=True,
                    use_container_width=True,
                )
            elif submission["analysis_error"]:
                st.error(
                    "Could not analyze the spoken feedback: "
                    f"{submission['analysis_error']}"
                )
            else:
                st.info("No clear topics were detected in the spoken feedback.")

            st.markdown("**Voice transcript**")
            st.write(submission["transcript"])
            st.download_button(
                "Download transcript",
                data=submission["transcript"],
                file_name="feedback-transcript.txt",
                mime="text/plain",
            )
            st.caption(
                "The transcript is stored temporarily on the server until you "
                "clear this session or the temporary file is cleaned up."
            )


if __name__ == "__main__":
    main()