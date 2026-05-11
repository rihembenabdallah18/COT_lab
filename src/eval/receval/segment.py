"""Sentence segmentation using spaCy.

Requires: python -m spacy download en_core_web_sm
"""
from __future__ import annotations

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy  # noqa: PLC0415
            _nlp = spacy.load("en_core_web_sm")
        except ImportError as exc:
            raise RuntimeError(
                "spaCy is not installed. Run: pip install spacy"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            ) from exc
    return _nlp


def segment(text: str) -> list[str]:
    """Split text into non-empty sentences."""
    if not text or not text.strip():
        return []
    nlp = _get_nlp()
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]
