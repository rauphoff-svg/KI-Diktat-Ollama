#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KI-Diktat-Ollama – Vollständig lokale Diktierlösung
====================================================
Push-to-Talk Aufnahme -> FasterWhisper (lokal) -> Ollama-Korrektur -> DOCX/TXT

Kein Datenaustausch mit externen Servern. Geeignet für Anwaltsdiktat
unter strengen Datenschutzanforderungen (DSGVO, Anwaltsgeheimnis).

Backends:
  Transkription: FasterWhisper (lokal, Modell konfigurierbar)
  Korrektur:     Ollama (lokal) – mistral-small:latest oder gemma3:latest

Aenderungsprotokoll v1:
  - Aufbau auf Basis von KI-Diktat-Voxtral-Claude v4.2
  - Transkription: Voxtral -> FasterWhisper (lokal)
  - Korrektur: Mistral/Claude -> Ollama (lokal)
  - Backend-Wechsel: mistral-small <-> gemma3

Abhaengigkeiten:
  pip install faster-whisper numpy sounddevice soundfile pyperclip pynput pyyaml python-docx
  Ollama muss lokal laufen (https://ollama.com)
"""

import io
import os
import sys
import json
import time
import queue
import threading
import re
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict

import yaml
import numpy as np
import sounddevice as sd
import soundfile as sf
import pyperclip
from pynput import keyboard

# Windows-Terminal: stdout auf UTF-8 umstellen
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# FasterWhisper
try:
    from faster_whisper import WhisperModel
    _HAS_FASTER_WHISPER = True
except ImportError:
    _HAS_FASTER_WHISPER = False

# python-docx (optional)
try:
    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    _HAS_DOCX = True
except ImportError:
    _HAS_DOCX = False

# pyperclip (optional)
try:
    import pyperclip as _pyperclip
    _HAS_PYPERCLIP = True
except ImportError:
    _HAS_PYPERCLIP = False


# =========================================================================
# TypedDicts
# =========================================================================

class AudioConfig(TypedDict):
    sample_rate: int
    channels: int
    blocksize: int
    silence_threshold_rms: float


class TranscriptionConfig(TypedDict):
    model: str
    language: str
    device: str
    compute_type: str


class CorrectionConfig(TypedDict):
    model: str
    ollama_base_url: str
    max_tokens: int
    temperature: float
    paragraph_wise: bool


class OutputConfig(TypedDict):
    format: str
    directory: str
    docx_font: str
    docx_font_size_pt: int
    docx_template: Optional[str]


class ControlsConfig(TypedDict):
    push_to_talk_key: str
    pause_key: str
    reject_segment_key: str
    segment_review: bool


class AppConfig(TypedDict):
    audio: AudioConfig
    transcription: TranscriptionConfig
    correction: CorrectionConfig
    output: OutputConfig
    controls: ControlsConfig


# =========================================================================
# Konfiguration
# =========================================================================

_DEFAULTS: dict = {
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "blocksize": 2048,
        "silence_threshold_rms": 0.005,
    },
    "transcription": {
        "model": "medium",
        "language": "de",
        "device": "cpu",
        "compute_type": "int8",
    },
    "correction": {
        "model": "mistral-small:latest",
        "ollama_base_url": "http://localhost:11434",
        "max_tokens": 4096,
        "temperature": 0.1,
        "paragraph_wise": True,
    },
    "output": {
        "format": "txt",
        "directory": ".",
        "docx_font": "Arial",
        "docx_font_size_pt": 11,
        "docx_template": None,
    },
    "controls": {
        "push_to_talk_key": "ctrl_r",
        "pause_key": "f10",
        "reject_segment_key": "f9",
        "segment_review": True,
    },
}

AVAILABLE_MODELS = ["mistral-small:latest", "gemma3:latest"]


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config() -> AppConfig:
    candidates = [
        Path(__file__).parent / "config.yaml",
        Path.cwd() / "config.yaml",
    ]
    cfg_path = next((p for p in candidates if p.is_file()), candidates[0])
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        return _deep_merge(_DEFAULTS, user_cfg)  # type: ignore
    return dict(_DEFAULTS)  # type: ignore


# =========================================================================
# Korrektur-Prompt
# =========================================================================

CORRECTION_SYSTEM_PROMPT = """\
Du bist ein juristischer Lektor. Deine Aufgabe ist ausschliesslich die \
sprachliche und formale Korrektur des diktierten Textes.

WICHTIGSTE REGEL: Du darfst NICHTS inhaltlich hinzufuegen!
- KEINE Paragraphen, Normen oder Gesetzeszitate ergaenzen, die nicht diktiert wurden
- KEINE juristischen Argumente, Begruendungen oder Erlaeuterungen einfuegen
- KEINE Sachverhaltselemente, Daten oder Fakten hinzudichten
- Der diktierte Inhalt ist die einzige Quelle – du korrigierst NUR Sprache und Form

Korrekturregeln:
1. Ausgesprochene Satzzeichen umwandeln:
   - "Komma" -> ,
   - "Punkt" / "Satzpunkt" -> .
   - "Doppelpunkt" -> :
   - "Semikolon" -> ;
   - "Gedankenstrich" -> \u2013 (mit Leerzeichen davor und danach)
   - "Bindestrich" -> -
   - "Schraegstrich" -> /
   - "Ausrufezeichen" -> !
   - "Fragezeichen" -> ?
   - "Klammer auf" -> (
   - "Klammer zu" -> )
   - "Anfuehrungszeichen auf" -> \u201e
   - "Anfuehrungszeichen zu" / "Anfuehrungszeichen Ende" -> \u201c
2. Normzitate: "Paragraph 823 Absatz 2 Satz 1 BGB" -> "\u00a7 823 Abs. 2 S. 1 BGB". \
Bereits diktierte Zitate korrekt formatieren, KEINE neuen Normen hinzufuegen.
3. Abkuerzungen: Nur standardisierte juristische Abkuerzungen (vgl., i.V.m., i.S.d.)
4. Stil:
   - Direkt und verstaendlich formulieren, NICHT uebermassig passivieren
   - Aktivsaetze bevorzugen
   - Klare, lebhafte Sprache – kein Kanzleideutsch
5. Interpunktion: Korrekte deutsche Zeichensetzung
6. Grammatik: Korrekte Deklination, Konjugation, Kongruenz
7. Rechtschreibung: Korrekte deutsche Rechtschreibung (Duden)
8. Absaetze: Erhalte Absatzstruktur exakt wie diktiert
9. Steuerkommandos:
   - "Seitenumbruch" oder "neue Seite" -> [SEITENUMBRUCH]
   - "Aufzaehlung:" -> echte Aufzaehlung mit Spiegelstrichen
   - "neuer Absatz" -> Absatzumbruch
10. Waehrung: Immer "EUR"
11. Prozent: Immer Zahl + Leerzeichen + % (z.B. "50 %")
12. Doppelwoerter aus Transkriptionsfehlern entfernen (z.B. "der der Klaeger" -> "der Klaeger")

Antworte NUR mit dem korrigierten Text, ohne Erklaerungen oder Kommentare.\
"""

CORRECTION_PARAGRAPH_PREFIX = """\
Vorheriger Absatz (nur als Kontext, nicht nochmals ausgeben):
{context}

Zu korrigierender Absatz:
{text}"""

CORRECTION_FULL_PREFIX = "{text}"


# =========================================================================
# Hilfsfunktionen
# =========================================================================

def _call_ollama(
    base_url: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout: int = 120,
) -> str:
    """Ruft die Ollama /api/chat Schnittstelle auf."""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode("utf-8"))
    return resp["message"]["content"].strip()


def _check_ollama(base_url: str, timeout: int = 3) -> bool:
    """Prüft ob Ollama erreichbar ist."""
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout)
        return True
    except Exception:
        return False


def _list_ollama_models(base_url: str) -> list[str]:
    """Listet verfügbare Ollama-Modelle."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def choose_input_device() -> int:
    """Lässt den Nutzer ein Eingabegerät wählen."""
    devices = sd.query_devices()
    inputs = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    if not inputs:
        raise RuntimeError("Kein Mikrofon gefunden.")
    print("\n  Verfügbare Mikrofone:")
    for i, d in inputs:
        print(f"    [{i}] {d['name']}")
    default_idx = sd.default.device[0]
    default_name = devices[default_idx]["name"] if default_idx is not None else ""
    print(f"\n  Standard: [{default_idx}] {default_name}")
    raw = input("  Auswahl (Enter = Standard): ").strip()
    if not raw:
        return default_idx
    idx = int(raw)
    if idx not in [i for i, _ in inputs]:
        raise ValueError(f"Ungültige Auswahl: {idx}")
    return idx


# =========================================================================
# Transkription mit FasterWhisper
# =========================================================================

class Transcriber:
    """Wrapper um FasterWhisper mit Lazy-Loading."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._model: Optional["WhisperModel"] = None

    def _load(self):
        if self._model is not None:
            return
        if not _HAS_FASTER_WHISPER:
            sys.exit(
                "FEHLER: faster-whisper nicht installiert.\n"
                "  pip install faster-whisper"
            )
        t_cfg = self.cfg["transcription"]
        print(f"  Lade Whisper-Modell '{t_cfg['model']}' "
              f"({t_cfg['device']}/{t_cfg['compute_type']}) …", end=" ", flush=True)
        self._model = WhisperModel(
            t_cfg["model"],
            device=t_cfg["device"],
            compute_type=t_cfg["compute_type"],
        )
        print("OK")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transkribiert ein Audio-Array."""
        self._load()
        # FasterWhisper erwartet float32 mono, 16kHz
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0

        lang = self.cfg["transcription"]["language"]
        segments, _ = self._model.transcribe(
            audio,
            language=lang,
            beam_size=5,
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


# =========================================================================
# DictationSession
# =========================================================================

class DictationSession:
    """Eine einzelne Diktiersitzung."""

    def __init__(self, cfg: AppConfig, transcriber: Transcriber):
        self.cfg = cfg
        self.transcriber = transcriber

        self.audio_queue: queue.Queue = queue.Queue()
        self.segments: list[str] = []
        self._recording = False
        self._paused = False
        self._stop = False
        self._processing_lock = threading.Lock()

        # Tastenbelegung
        ctrl = self.cfg["controls"]
        self._ptk = ctrl["push_to_talk_key"].lower()
        self._pause_key = ctrl["pause_key"].lower()
        self._reject_key = ctrl["reject_segment_key"].lower()

    # ------------------------------------------------------------------
    # Tastatur
    # ------------------------------------------------------------------

    def _key_name(self, key) -> str:
        try:
            return key.char.lower() if key.char else ""
        except AttributeError:
            return str(key).lower().replace("key.", "")

    def _on_press(self, key):
        name = self._key_name(key)
        if name == "esc":
            self._stop = True
            return False
        if name == self._pause_key:
            self._paused = not self._paused
            state = "PAUSE" if self._paused else "WEITER"
            print(f"\n  [{state}]", flush=True)
        if name == self._reject_key:
            if self.segments:
                removed = self.segments.pop()
                print(f"\n  [Verworfen] {removed[:60]}…" if len(removed) > 60
                      else f"\n  [Verworfen] {removed}", flush=True)

    def _on_release(self, key):
        pass

    # ------------------------------------------------------------------
    # Audio-Aufnahme
    # ------------------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):
        if self._recording and not self._paused:
            self.audio_queue.put(indata.copy())

    def _rms(self, data: np.ndarray) -> float:
        return float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))

    def _collect_audio(self) -> np.ndarray:
        chunks = []
        while not self.audio_queue.empty():
            chunks.append(self.audio_queue.get_nowait())
        return np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=np.int16)

    # ------------------------------------------------------------------
    # Segmentverarbeitung
    # ------------------------------------------------------------------

    def _process_segment(self, audio: np.ndarray):
        """Transkribiert und korrigiert ein Audiosegment."""
        sr = self.cfg["audio"]["sample_rate"]
        threshold = self.cfg["audio"]["silence_threshold_rms"]

        if self._rms(audio) < threshold:
            return

        with self._processing_lock:
            print(f"\n  Transkribiere ({len(audio)/sr:.1f}s) …", flush=True)
            text = self.transcriber.transcribe(audio, sr)
            if not text:
                print("  (kein Text erkannt)")
                return
            print(f"  -> {text}")

            if self.cfg["controls"]["segment_review"]:
                print("  [Enter=Übernehmen  F9=Verwerfen] ", end="", flush=True)

            self.segments.append(text)

            if self.cfg["controls"]["segment_review"]:
                # Kurz warten ob F9 gedrückt wird
                time.sleep(0.8)
                if self.segments and self.segments[-1] == text:
                    print("Übernommen")
                # (F9-Handler hat ggf. bereits verworfen)

    # ------------------------------------------------------------------
    # Push-to-Talk Hauptschleife
    # ------------------------------------------------------------------

    def _print_controls(self):
        model = self.cfg["correction"]["model"]
        ctrl = self.cfg["controls"]
        print(f"\n  Korrektur-KI: Ollama ({model})")
        print(f"  [{ctrl['push_to_talk_key'].upper()}]=Aufnahme  "
              f"[{ctrl['reject_segment_key'].upper()}]=Verwerfen  "
              f"[{ctrl['pause_key'].upper()}]=Pause  [ESC]=Beenden")

    def run(self, device_idx: int):
        """Startet die Diktiersitzung."""
        sr = self.cfg["audio"]["sample_rate"]
        bs = self.cfg["audio"]["blocksize"]

        self._print_controls()

        # Prüfe ob Push-to-Talk ctrl_r oder einfache Taste ist
        ptk = self._ptk  # z.B. "ctrl_r" oder "f8"
        is_ctrl_combo = ptk.startswith("ctrl_")
        trigger_key = ptk.split("_")[1] if is_ctrl_combo else ptk

        recording_buffer: list[np.ndarray] = []
        ctrl_held = False
        trigger_held = False

        def on_press(key):
            nonlocal ctrl_held, trigger_held
            name = self._key_name(key)

            # ESC
            if name == "esc":
                self._stop = True
                return False

            # Pause
            if name == self._pause_key:
                self._paused = not self._paused
                state = "PAUSE" if self._paused else "WEITER"
                print(f"\n  [{state}]", flush=True)
                return

            # Verwerfen
            if name == self._reject_key and self.segments:
                removed = self.segments.pop()
                short = removed[:60] + "…" if len(removed) > 60 else removed
                print(f"\n  [Verworfen] {short}", flush=True)
                return

            # Push-to-Talk
            if is_ctrl_combo:
                if name in ("ctrl_l", "ctrl_r", "ctrl"):
                    ctrl_held = True
                if name == trigger_key and ctrl_held and not self._recording:
                    self._recording = True
                    print("\n  ■ AUFNAHME …", end=" ", flush=True)
            else:
                if name == trigger_key and not self._recording:
                    self._recording = True
                    print("\n  ■ AUFNAHME …", end=" ", flush=True)

        def on_release(key):
            nonlocal ctrl_held, trigger_held
            name = self._key_name(key)

            if is_ctrl_combo:
                if name in ("ctrl_l", "ctrl_r", "ctrl"):
                    ctrl_held = False
                    if self._recording:
                        self._recording = False
                        audio = self._collect_audio()
                        if len(audio) > sr * 0.3:
                            t = threading.Thread(
                                target=self._process_segment, args=(audio,), daemon=True
                            )
                            t.start()
                        else:
                            print("  (zu kurz)")
                if name == trigger_key:
                    trigger_held = False
            else:
                if name == trigger_key and self._recording:
                    self._recording = False
                    audio = self._collect_audio()
                    if len(audio) > sr * 0.3:
                        t = threading.Thread(
                            target=self._process_segment, args=(audio,), daemon=True
                        )
                        t.start()
                    else:
                        print("  (zu kurz)")

        with sd.InputStream(
            device=device_idx,
            channels=self.cfg["audio"]["channels"],
            samplerate=sr,
            blocksize=bs,
            dtype="int16",
            callback=self._audio_callback,
        ):
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                listener.join()

        # Warte auf laufende Verarbeitung
        self._processing_lock.acquire()
        self._processing_lock.release()

        self._finalize()

    # ------------------------------------------------------------------
    # Abschluss: Korrektur und Ausgabe
    # ------------------------------------------------------------------

    def _finalize(self):
        print(f"\n  [ESC] Diktat beenden …")
        if not self.segments:
            print("  Kein Text diktiert.")
            return

        raw_text = "\n\n".join(self.segments)
        print(f"\n{'=' * 60}")
        print(f"  ROHTEXT ({len(self.segments)} Segment{'e' if len(self.segments) != 1 else ''})")
        print(f"{'–' * 60}")
        print(raw_text)
        print(f"{'–' * 60}")

        # Korrektur
        corrected = self._correct(raw_text)

        print(f"\n{'=' * 60}")
        print("  KORRIGIERTER TEXT")
        print(f"{'–' * 60}")
        print(corrected)
        print(f"{'=' * 60}")

        # In Zwischenablage
        if _HAS_PYPERCLIP:
            try:
                _pyperclip.copy(corrected)
                print("  -> In Zwischenablage kopiert")
            except Exception:
                pass

        # Speichern
        self._save(corrected)

    def _correct(self, text: str) -> str:
        """Korrigiert den Text via Ollama."""
        cfg = self.cfg["correction"]
        base_url = cfg["ollama_base_url"]
        model = cfg["model"]
        temp = cfg.get("temperature", 0.1)
        max_tokens = cfg.get("max_tokens", 4096)
        paragraph_wise = cfg.get("paragraph_wise", True)

        print(f"\n  Korrektur mit Ollama ({model}):")

        if paragraph_wise:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            corrected = []
            prev_context = ""
            for i, para in enumerate(paragraphs):
                print(f"    Absatz {i+1}/{len(paragraphs)} …", end=" ", flush=True)
                if prev_context:
                    prompt = CORRECTION_PARAGRAPH_PREFIX.format(
                        context=prev_context[-500:], text=para
                    )
                else:
                    prompt = CORRECTION_FULL_PREFIX.format(text=para)

                result = self._call_with_retry(base_url, model, prompt, temp, max_tokens)
                corrected.append(result)
                prev_context = result
                print("OK")
            return "\n\n".join(corrected)
        else:
            print("    …", end=" ", flush=True)
            result = self._call_with_retry(
                base_url, model,
                CORRECTION_FULL_PREFIX.format(text=text),
                temp, max_tokens,
            )
            print("OK")
            return result

    def _call_with_retry(
        self,
        base_url: str,
        model: str,
        user_prompt: str,
        temp: float,
        max_tokens: int,
        max_retries: int = 3,
    ) -> str:
        for attempt in range(1, max_retries + 1):
            try:
                return _call_ollama(
                    base_url, model,
                    CORRECTION_SYSTEM_PROMPT, user_prompt,
                    temperature=temp, max_tokens=max_tokens,
                )
            except Exception as exc:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"\n  [Versuch {attempt}/{max_retries} fehlgeschlagen: {exc}]"
                          f"\n  Warte {wait}s …", file=sys.stderr)
                    time.sleep(wait)
                else:
                    print(f"\n  [Fehler nach {max_retries} Versuchen: {exc}]",
                          file=sys.stderr)
                    return user_prompt  # Rohtext als Fallback
        return user_prompt

    def _save(self, text: str):
        out_cfg = self.cfg["output"]
        out_dir = Path(out_cfg["directory"])
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fmt = out_cfg["format"].lower()

        if fmt == "docx" and _HAS_DOCX:
            path = out_dir / f"diktat_{ts}.docx"
            self._save_docx(text, path, out_cfg)
        else:
            path = out_dir / f"diktat_{ts}.txt"
            path.write_text(text, encoding="utf-8")

        print(f"  Gespeichert: {path.resolve()}")

    def _save_docx(self, text: str, path: Path, cfg: dict):
        template = cfg.get("docx_template")
        if template and Path(template).is_file():
            doc = Document(template)
        else:
            doc = Document()

        font_name = cfg.get("docx_font", "Arial")
        font_size = cfg.get("docx_font_size_pt", 11)

        for block in text.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            if block == "[SEITENUMBRUCH]":
                doc.add_page_break()
                continue
            p = doc.add_paragraph()
            run = p.add_run(block)
            run.font.name = font_name
            run.font.size = Pt(font_size)

        doc.save(path)


# =========================================================================
# Hauptprogramm
# =========================================================================

def main():
    if not _HAS_FASTER_WHISPER:
        sys.exit(
            "FEHLER: faster-whisper nicht installiert.\n"
            "  pip install faster-whisper"
        )

    cfg = load_config()
    base_url = cfg["correction"]["ollama_base_url"]

    print("=" * 60)
    print("  KI-Diktat-Ollama – Vollständig lokale Lösung")
    print("=" * 60)

    # Ollama prüfen
    if not _check_ollama(base_url):
        sys.exit(
            f"FEHLER: Ollama nicht erreichbar ({base_url})\n"
            "  Starte Ollama: ollama serve"
        )

    # Verfügbare Modelle anzeigen
    available = _list_ollama_models(base_url)
    print(f"\n  Ollama erreichbar: {base_url}")
    if available:
        print(f"  Verfügbare Modelle: {', '.join(available)}")

    # Aktuelles Modell prüfen
    current_model = cfg["correction"]["model"]
    if available and current_model not in available:
        print(f"\n  [Warnung] Modell '{current_model}' nicht in Ollama verfügbar.")
        print(f"  Tipp: ollama pull {current_model}")

    # Transkriptions-Modell
    t_cfg = cfg["transcription"]
    print(f"\n  Transkription: FasterWhisper '{t_cfg['model']}' "
          f"({t_cfg['device']}/{t_cfg['compute_type']})")
    print(f"  Korrektur:     Ollama ({current_model})")

    # Mikrofon wählen
    try:
        device_idx = choose_input_device()
    except (RuntimeError, ValueError) as exc:
        sys.exit(f"FEHLER: {exc}")

    dev_info = sd.query_devices(device_idx)
    print(f"\n  Verwende: [{device_idx}] {dev_info['name']}")

    # Transcriber einmal erstellen (Modell wird beim ersten Diktat geladen)
    transcriber = Transcriber(cfg)

    # Session-Schleife
    diktat_nr = 0
    while True:
        diktat_nr += 1

        model = cfg["correction"]["model"]
        if diktat_nr > 1:
            print(f"\n{'=' * 60}")
            print(f"  Neues Diktat #{diktat_nr}  (Ollama: {model})")
            print(f"{'=' * 60}")

        session = DictationSession(cfg, transcriber)
        session.run(device_idx)

        # Menü nach Diktat
        while True:
            model = cfg["correction"]["model"]
            print()
            print("  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄")
            print(f"  Korrektur-KI: Ollama ({model})")
            print("  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄")
            print("  [Enter]     Neues Diktat")
            print("  [B + Enter] Modell wechseln  "
                  "(mistral-small <-> gemma3)")
            print("  [Q + Enter] Tool beenden")
            try:
                choice = input("  > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "q"

            if choice in ("b", "modell", "wechseln"):
                # Reihenfolge: mistral-small:latest <-> gemma3:latest
                order = AVAILABLE_MODELS
                current = cfg["correction"]["model"]
                next_idx = (order.index(current) + 1) % len(order) if current in order else 0
                next_model = order[next_idx]

                # Verfügbarkeit prüfen
                if available and next_model not in available:
                    print(f"\n  [!] '{next_model}' nicht in Ollama – "
                          f"Tipp: ollama pull {next_model}")
                else:
                    cfg["correction"]["model"] = next_model
                    print(f"\n  -> Gewechselt: Ollama ({next_model})")
                continue
            else:
                break

        if choice in ("q", "quit", "exit", "beenden"):
            break

    print("\n  Diktiertool beendet. Auf Wiedersehen!")


if __name__ == "__main__":
    main()
