"""CLI for converting audio files using ffmpeg."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional


def setup_logging(debug: bool = False, log_path: Optional[str] = None) -> None:
    """Configure logging based on arguments."""
    level = logging.DEBUG if debug else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler(sys.stderr)]
    if log_path:
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(level=level, format=format_str, handlers=handlers)


def convert_audio(
    input_path: str,
    output_path: Optional[str] = None,
    input_type: Optional[str] = None,
    output_type: Optional[str] = None,
    bitrate: Optional[str] = None,
    quality: Optional[int] = None,
    debug: bool = False,
) -> int:
    """
    Convert audio file using ffmpeg.

    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file (optional)
        input_type: Force input container format
        output_type: Force output container format
        bitrate: Bitrate (e.g. 128k). Overrides quality if present.
        quality: VBR quality (0..9) for encoders that support it (0=best)
        debug: Enable debug logging and stream ffmpeg output

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger = logging.getLogger(__name__)

    # Determine output path if not provided
    if output_path is None:
        input_file = Path(input_path)
        if output_type:
            output_path = str(input_file.with_suffix(f".{output_type}"))
        else:
            output_path = str(input_file.with_suffix(".mp3"))
        logger.info(f"Output path not specified, using: {output_path}")

    # Build ffmpeg command
    cmd = ["ffmpeg", "-i", input_path]

    # Add input format if specified
    if input_type:
        cmd.insert(1, "-f")
        cmd.insert(2, input_type)

    # Add audio codec and quality settings
    if bitrate:
        cmd.extend(["-b:a", bitrate])
        logger.debug(f"Using bitrate: {bitrate}")
    elif quality is not None:
        cmd.extend(["-q:a", str(quality)])
        logger.debug(f"Using VBR quality: {quality}")

    # Add output format if specified
    if output_type:
        cmd.extend(["-f", output_type])

    # Add output path
    cmd.append(output_path)

    logger.info(f"Converting {input_path} to {output_path}")
    logger.debug(f"ffmpeg command: {' '.join(cmd)}")

    try:
        # Run ffmpeg
        if debug:
            # Stream output in debug mode
            result = subprocess.run(cmd, check=True)
        else:
            # Capture output in normal mode
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

        logger.info(f"Successfully converted to {output_path}")
        return 0

    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed with exit code {e.returncode}")
        if not debug and e.stderr:
            logger.error(f"ffmpeg stderr: {e.stderr}")
        return e.returncode
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please ensure ffmpeg is installed and on PATH.")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert and transcribe audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input file path (required)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path"
    )

    # Transcription options
    parser.add_argument(
        "-t", "--transcribe",
        action="store_true",
        help="Transcribe audio to text using Whisper (requires whisper extra)"
    )
    parser.add_argument(
        "-m", "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "-L", "--language",
        help="Language code for transcription (e.g., 'en', 'es'). Auto-detect if not specified."
    )
    parser.add_argument(
        "-f", "--format",
        default="txt",
        choices=["txt", "srt", "vtt", "json"],
        help="Transcription output format (default: txt)"
    )
    parser.add_argument(
        "--backend",
        default="faster-whisper",
        choices=["whisper", "faster-whisper", "whisperx"],
        help="Transcription backend (default: faster-whisper)"
    )
    parser.add_argument(
        "--stream-speed",
        type=float,
        default=30.0,
        help="Characters per second for visual text streaming (default: 30.0, 0 to disable)"
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization (whisperx only, requires --hf-token)"
    )
    parser.add_argument(
        "--hf-token",
        help="Hugging Face token for speaker diarization (or set HF_TOKEN env var)"
    )

    # Conversion options
    parser.add_argument(
        "-I", "--input-type",
        help="Force input container format (m4a, wav, ...)"
    )
    parser.add_argument(
        "-O", "--output-type",
        help="Force output container format (mp3, wav, ...)"
    )
    parser.add_argument(
        "-b", "--bitrate",
        help="Bitrate (e.g. 128k). Overrides quality if present."
    )
    parser.add_argument(
        "-q", "--quality",
        type=int,
        choices=range(10),
        metavar="0-9",
        help="VBR quality (0..9) for encoders that support it (0=best)."
    )

    # General options
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug logging and stream ffmpeg output."
    )
    parser.add_argument(
        "-l", "--log",
        help="Path to write log file."
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(debug=args.debug, log_path=args.log)

    # Transcribe if requested
    if args.transcribe:
        from .transcribe import transcribe_audio
        import os

        # Try to load .env file if python-dotenv is available
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass  # python-dotenv not installed, skip

        # Get HF token from args or environment (support both variable names)
        hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_USER_TOKEN")

        return transcribe_audio(
            input_path=args.input,
            model=args.model,
            language=args.language,
            output_format=args.format,
            output_path=args.output,
            backend=args.backend,
            stream_speed=args.stream_speed,
            diarize=args.diarize,
            hf_token=hf_token,
        )

    # Otherwise, convert audio
    return convert_audio(
        input_path=args.input,
        output_path=args.output,
        input_type=args.input_type,
        output_type=args.output_type,
        bitrate=args.bitrate,
        quality=args.quality,
        debug=args.debug,
    )


if __name__ == "__main__":
    sys.exit(main())
