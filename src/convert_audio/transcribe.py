"""Transcription functionality using OpenAI Whisper or faster-whisper."""

import logging
from pathlib import Path
from typing import Optional, Literal
import torch

# Detect if GPU is available
use_gpu = torch.cuda.is_available()
   # do_diarize comes from CLI flag

logger = logging.getLogger(__name__)

BackendType = Literal["whisper", "faster-whisper", "whisperx"]


def transcribe_audio(
    input_path: str,
    model: str = "base",
    language: Optional[str] = None,
    output_format: str = "txt",
    output_path: Optional[str] = None,
    backend: BackendType = "faster-whisper",
    stream_speed: float = 30.0,
    diarize: bool = False,
    hf_token: Optional[str] = None,
) -> int:
    """
    Transcribe audio file using OpenAI Whisper or faster-whisper.

    Args:
        input_path: Path to input audio file
        model: Whisper model size (tiny, base, small, medium, large)
        language: Language code (e.g., 'en', 'es'). Auto-detect if None.
        output_format: Output format (txt, srt, vtt, json)
        output_path: Path to output transcription file (optional)
        backend: Which backend to use ('whisper' or 'faster-whisper')
        stream_speed: Characters per second for visual streaming (default: 30.0)
        diarize: Enable speaker diarization (whisperx only)
        hf_token: Hugging Face token for diarization (required if diarize=True)

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    if backend == "faster-whisper":
        return transcribe_with_faster_whisper(
            input_path, model, language, output_format, output_path, stream_speed
        )
    elif backend == "whisperx":
        return transcribe_with_whisperx(
            input_path, model, language, output_format, output_path, stream_speed, diarize, hf_token
        )
    else:
        return transcribe_with_whisper(
            input_path, model, language, output_format, output_path
        )


def transcribe_with_whisper(
    input_path: str,
    model: str,
    language: Optional[str],
    output_format: str,
    output_path: Optional[str],
) -> int:
    """Transcribe using original openai-whisper."""
    try:
        import whisper
    except ImportError:
        logger.error(
            "Whisper is not installed. Install with: pip install -e .[whisper]"
        )
        return 1

    logger.info(f"Loading Whisper model '{model}'...")
    try:
        model_obj = whisper.load_model(model)
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        return 1

    logger.info(f"Transcribing {input_path}...")
    try:
        result = model_obj.transcribe(
            input_path,
            language=language,
            verbose=False,
        )
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return 1

    # Determine output path if not provided
    if output_path is None:
        input_file = Path(input_path)
        output_path = str(input_file.with_suffix(f".{output_format}"))
        logger.info(f"Output path not specified, using: {output_path}")

    # Write output based on format
    return write_output(result, output_format, output_path)


def transcribe_with_faster_whisper(
    input_path: str,
    model: str,
    language: Optional[str],
    output_format: str,
    output_path: Optional[str],
    stream_speed: float = 30.0,
) -> int:
    """Transcribe using faster-whisper (optimized, 4x faster)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.error(
            "faster-whisper is not installed. Install with: pip install -e .[faster-whisper]"
        )
        return 1

    logger.info(f"Loading faster-whisper model '{model}'...")
    try:
        # Use CPU with int8 for efficiency, or cuda if available
        model_obj = WhisperModel(model, device="cpu", compute_type="int8")
        logger.info("Using faster-whisper backend (4x faster than standard whisper)")
    except Exception as e:
        logger.error(f"Failed to load faster-whisper model: {e}")
        return 1

    logger.info(f"Transcribing {input_path}...")
    try:
        # Get audio duration for progress tracking
        try:
            import subprocess
            result_probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_path],
                capture_output=True,
                text=True,
                check=True
            )
            total_duration = float(result_probe.stdout.strip())
        except Exception:
            total_duration = None

        # faster-whisper returns segments and info
        segments, info = model_obj.transcribe(
            input_path,
            language=language,
            beam_size=5,
            vad_filter=True,  # Voice activity detection - skips silence
        )

        logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

        # Convert generator to list and build result dict with progress bar
        segments_list = []
        full_text = []

        # Try to use tqdm for visual progress
        try:
            from tqdm import tqdm
            import sys
            import time
            import threading

            # Create progress bar if we know duration
            if total_duration:
                pbar = tqdm(
                    total=total_duration,
                    unit='s',
                    desc='Transcribing',
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining}]',
                    file=sys.stderr,
                    position=0,
                    leave=True,
                    dynamic_ncols=True
                )
            else:
                pbar = tqdm(
                    desc='Transcribing',
                    unit=' segments',
                    bar_format='{desc}: {n} segments [{elapsed}]',
                    file=sys.stderr,
                    position=0,
                    leave=True,
                    dynamic_ncols=True
                )

            # Character-by-character streaming state (only if stream_speed > 0)
            if stream_speed > 0:
                max_line_length = 120  # Maximum characters per line
                current_line = ""  # Current line being built
                pending_text = ""  # Text waiting to be streamed
                stream_lock = threading.Lock()
                streaming_active = threading.Event()
                streaming_active.set()

                def stream_characters():
                    """Background thread to stream characters at constant rate."""
                    nonlocal current_line, pending_text
                    char_delay = 1.0 / stream_speed  # Delay between characters
                    last_display = ""

                    while streaming_active.is_set() or pending_text:
                        with stream_lock:
                            if pending_text:
                                # Add one character to current line
                                current_line += pending_text[0]
                                pending_text = pending_text[1:]

                                # Check if we need to wrap (break at word boundary)
                                if len(current_line) > max_line_length:
                                    # Find last space to break at word boundary
                                    last_space = current_line.rfind(' ', 0, max_line_length)

                                    if last_space > 0:
                                        # Break at last space
                                        line_to_print = current_line[:last_space]
                                        current_line = current_line[last_space+1:]  # Keep remainder
                                    else:
                                        # No space found, force break
                                        line_to_print = current_line
                                        current_line = ""

                                    # Print completed line using tqdm.write (handles positioning)
                                    pbar.write(line_to_print.strip())
                                    last_display = ""

                                # Update current line using set_postfix
                                display_text = current_line.strip()
                                if display_text != last_display:
                                    pbar.set_postfix_str(display_text[-80:] if len(display_text) > 80 else display_text)
                                    last_display = display_text

                        time.sleep(char_delay)

                # Start streaming thread
                stream_thread = threading.Thread(target=stream_characters, daemon=True)
                stream_thread.start()

            for segment in segments:
                segments_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                })
                full_text.append(segment.text)

                # Add segment text to pending queue if streaming enabled
                if stream_speed > 0:
                    with stream_lock:
                        pending_text += segment.text

                # Update progress bar
                if total_duration:
                    pbar.n = segment.end
                    pbar.refresh()
                else:
                    pbar.update(1)

            # Wait for all pending text to stream (if streaming enabled)
            if stream_speed > 0:
                while pending_text:
                    time.sleep(0.1)

                # Stop streaming thread
                streaming_active.clear()
                stream_thread.join(timeout=2)

                # Clear the postfix
                pbar.set_postfix_str("")

            # Complete the progress bar
            if total_duration:
                pbar.n = total_duration
                pbar.refresh()
            pbar.close()

        except ImportError:
            # Fallback to simple logging if tqdm not available
            for segment in segments:
                segments_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                })
                full_text.append(segment.text)

                # Show progress
                if total_duration:
                    progress = min((segment.end / total_duration) * 100, 100)
                    logger.info(f"Progress: {progress:.1f}% [{segment.end:.1f}s / {total_duration:.1f}s]: {segment.text.strip()}")
                else:
                    logger.info(f"Processed: {segment.end:.1f}s: {segment.text.strip()}")

        result = {
            "text": " ".join(full_text),
            "segments": segments_list,
            "language": info.language,
        }

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return 1

    # Determine output path if not provided
    if output_path is None:
        input_file = Path(input_path)
        output_path = str(input_file.with_suffix(f".{output_format}"))
        logger.info(f"Output path not specified, using: {output_path}")

    # Write output based on format
    return write_output(result, output_format, output_path)


def transcribe_with_whisperx(
    input_path: str,
    model: str,
    language: Optional[str],
    output_format: str,
    output_path: Optional[str],
    stream_speed: float = 30.0,
    diarize: bool = False,
    hf_token: Optional[str] = None,
) -> int:
    """Transcribe using WhisperX (includes speaker diarization and word-level timestamps)."""
    try:
        import whisperx
    except ImportError:
        logger.error(
            "WhisperX is not installed. Install with: pip install -e .[whisperx]"
        )
        return 1

    # Check diarization requirements

       # do_diarize comes from CLI flag
    if not use_gpu and diarize:
        print("GPU not detected. Skipping speaker diarization for speed and compatibility.")
        diarize = False

    if diarize and not hf_token:
        logger.error(
            "Speaker diarization requires a Hugging Face token. "
            "Provide it via --hf-token or HF_TOKEN environment variable."
        )
        return 1

    logger.info(f"Loading WhisperX model '{model}'...")
    try:
        # WhisperX uses device and compute_type similar to faster-whisper
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        model_obj = whisperx.load_model(model, device, compute_type=compute_type)
        diarize_str = " with diarization" if diarize and do_diarize else ""
        logger.info(f"Using WhisperX backend on {device} (word-level timestamps{diarize_str})")
    except Exception as e:
        logger.error(f"Failed to load WhisperX model: {e}")
        return 1

    logger.info(f"Transcribing {input_path}...")
    try:
        # Load audio
        audio = whisperx.load_audio(input_path)

        # Get audio duration for progress tracking
        try:
            import subprocess
            result_probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_path],
                capture_output=True,
                text=True,
                check=True
            )
            total_duration = float(result_probe.stdout.strip())
        except Exception:
            total_duration = None

        # Transcribe with WhisperX
        result = model_obj.transcribe(audio, language=language, batch_size=16)

        detected_language = result.get("language", "unknown")
        logger.info(f"Detected language: {detected_language}")

        # Align whisper output for word-level timestamps
        if diarize:
            logger.info("Aligning transcript for word-level timestamps...")
            model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

            # Perform speaker diarization
            logger.info("Performing speaker diarization...")
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            logger.info("Speaker diarization complete")

        # Extract segments
        segments_list = []
        full_text = []

        for segment in result.get("segments", []):
            seg_dict = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
            }
            # Add speaker info if available
            if "speaker" in segment:
                seg_dict["speaker"] = segment["speaker"]

            segments_list.append(seg_dict)

            # Format text with speaker labels if available
            if "speaker" in segment:
                full_text.append(f"[{segment['speaker']}] {segment['text']}")
            else:
                full_text.append(segment["text"])

        result_dict = {
            "text": " ".join(full_text),
            "segments": segments_list,
            "language": detected_language,
        }

        logger.info(f"Transcribed {len(segments_list)} segments")

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return 1

    # Determine output path if not provided
    if output_path is None:
        input_file = Path(input_path)
        output_path = str(input_file.with_suffix(f".{output_format}"))
        logger.info(f"Output path not specified, using: {output_path}")

    # Write output based on format
    return write_output(result_dict, output_format, output_path)


def write_output(result: dict, output_format: str, output_path: str) -> int:
    """Write transcription result to file."""
    try:
        output_file = Path(output_path)

        if output_format == "txt":
            output_file.write_text(result["text"].strip())
            logger.info(f"Transcription saved to {output_path}")

        elif output_format == "srt":
            write_srt(result["segments"], output_file)
            logger.info(f"SRT subtitles saved to {output_path}")

        elif output_format == "vtt":
            write_vtt(result["segments"], output_file)
            logger.info(f"VTT subtitles saved to {output_path}")

        elif output_format == "json":
            import json
            output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            logger.info(f"JSON transcription saved to {output_path}")

        else:
            logger.error(f"Unsupported output format: {output_format}")
            return 1

        # Print transcription to console
        logger.info(f"\nTranscription:\n{result['text'].strip()}\n")

        return 0

    except Exception as e:
        logger.error(f"Failed to write output: {e}")
        return 1


def write_srt(segments: list, output_file: Path) -> None:
    """Write segments as SRT subtitle format."""
    lines = []
    for i, segment in enumerate(segments, start=1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    output_file.write_text("\n".join(lines))


def write_vtt(segments: list, output_file: Path) -> None:
    """Write segments as WebVTT subtitle format."""
    lines = ["WEBVTT\n"]
    for segment in segments:
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()
        lines.append(f"{start} --> {end}\n{text}\n")
    output_file.write_text("\n".join(lines))


def format_timestamp(seconds: float) -> str:
    """Format seconds as timestamp (HH:MM:SS,mmm for SRT / HH:MM:SS.mmm for VTT)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
