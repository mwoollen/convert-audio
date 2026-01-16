# convert-audio

Small CLI to convert audio files (e.g. `.m4a`) to other formats (e.g. `.mp3`) using `ffmpeg`, and transcribe audio to text using OpenAI Whisper.

Requirements
- Python 3.12+
- `ffmpeg` available on PATH (for audio conversion)
- OpenAI Whisper (optional, for transcription - install with `.[whisper]` extra)

Installing ffmpeg
```bash
# Ubuntu/Debian/WSL
sudo apt update && sudo apt install ffmpeg

# macOS (with Homebrew)
brew install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg

# Verify installation
ffmpeg -version
```

Install convert-audio (editable/dev)
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
python -m pip install -e .
```

Install convert-audio (regular)
```bash
python -m pip install .
```

Install with Whisper transcription support
```bash
# Install with faster-whisper (recommended, 4x faster, less memory)
python -m pip install -e .[faster-whisper]

# Or install with WhisperX (word-level timestamps, speaker diarization, GPU support)
python -m pip install -e .[whisperx]

# Or install with original OpenAI Whisper
python -m pip install -e .[whisper]

# Install with dev dependencies
python -m pip install -e .[faster-whisper,dev]
```

## Usage

### Audio Conversion
Convert audio files between formats:
```bash
# Basic conversion (m4a to mp3)
convert-audio -i input.m4a -o output.mp3

# Specify output format
convert-audio -i input.m4a -O mp3

# Set bitrate
convert-audio -i input.wav -o output.mp3 -b 192k

# Set VBR quality
convert-audio -i input.wav -o output.mp3 -q 2
```

### Audio Transcription
Transcribe audio to text using Whisper:
```bash
# Basic transcription (uses faster-whisper by default, outputs input.txt)
convert-audio -i audio.m4a -t

# Specify output file
convert-audio -i audio.m4a -t -o transcript.txt

# Use different Whisper model (tiny, base, small, medium, large)
convert-audio -i audio.m4a -t -m small

# Specify language (faster and more accurate)
convert-audio -i audio.m4a -t -L en

# Use original OpenAI Whisper backend (slower)
convert-audio -i audio.m4a -t --backend whisper

# Use WhisperX backend (word-level timestamps, diarization)
convert-audio -i audio.m4a -t --backend whisperx

# Enable speaker diarization with WhisperX (requires HF token)
convert-audio -i audio.m4a -t --backend whisperx --diarize --hf-token YOUR_HF_TOKEN

# Generate SRT subtitles
convert-audio -i audio.m4a -t -f srt -o subtitles.srt

# Generate VTT subtitles
convert-audio -i audio.m4a -t -f vtt

# Generate JSON output with timestamps
convert-audio -i audio.m4a -t -f json
```

### Whisper Backends
- **faster-whisper** (default): 4x faster than standard whisper, uses less memory, includes VAD (voice activity detection) to skip silence, shows live progress bar and character-by-character transcription feed that wraps to new lines
- **whisperx**: Enhanced version with word-level timestamps, speaker diarization support, and optimized GPU/CPU performance
- **whisper**: Original OpenAI implementation

### Speaker Diarization (WhisperX only)
Speaker diarization identifies and labels different speakers in the audio. To use it:

1. Get a Hugging Face token:
   - Create account at https://huggingface.co
   - Go to Settings → Access Tokens
   - Create a token with read access
   - Accept the pyannote/segmentation model terms at https://huggingface.co/pyannote/segmentation

2. Use diarization:
```bash
# Pass token via command line
convert-audio -i meeting.m4a -t --backend whisperx --diarize --hf-token YOUR_TOKEN

# Or set environment variable (either name works)
export HF_TOKEN=YOUR_TOKEN
convert-audio -i meeting.m4a -t --backend whisperx --diarize

# Or create a .env file in your project directory (either variable name works)
echo "HF_TOKEN=YOUR_TOKEN" > .env
# or
echo "HUGGINGFACE_USER_TOKEN=YOUR_TOKEN" > .env
convert-audio -i meeting.m4a -t --backend whisperx --diarize
```

Output will include speaker labels like `[SPEAKER_00] Hello everyone` in the transcript.

**Security Note:** The `.env` file is automatically excluded from git via `.gitignore` to keep your token secure. Never commit your token to version control.

### Whisper Models
- **tiny**: Fastest, least accurate (~1GB RAM)
- **base**: Fast, good for most uses (~1GB RAM) - **default**
- **small**: Better accuracy (~2GB RAM)
- **medium**: High accuracy (~5GB RAM)
- **large**: Best accuracy, slowest (~10GB RAM)

**Performance Comparison (base model, 1 min audio):**
- faster-whisper: ~5-10 seconds
- standard whisper: ~20-40 seconds

### Options

#### General
- `-i, --input`: Input file path (required)
- `-o, --output`: Output file path
- `-d, --debug`: Enable debug logging
- `-l, --log`: Path to write log file

#### Transcription (requires `-t`)
- `-t, --transcribe`: Enable transcription mode
- `-m, --model`: Whisper model (tiny/base/small/medium/large, default: base)
- `-L, --language`: Language code (e.g., en, es, fr) for faster/better transcription
- `-f, --format`: Output format (txt/srt/vtt/json, default: txt)
- `--backend`: Backend to use (whisper/faster-whisper/whisperx, default: faster-whisper)
- `--stream-speed`: Characters per second for visual text streaming (default: 30.0, set to 0 to disable)
- `--diarize`: Enable speaker diarization (whisperx only)
- `--hf-token`: Hugging Face token for diarization (or set HF_TOKEN environment variable)

#### Audio Conversion
- `-I, --input-type`: Force input container format (m4a, wav, ...)
- `-O, --output-type`: Force output container format (mp3, wav, ...)
- `-b, --bitrate`: Bitrate (e.g. 128k). Overrides quality if present.
- `-q, --quality`: VBR quality (0..9) for encoders that support it (0=best)

Running tests
- Ensure ffmpeg is on PATH locally, then run:
  - python -m pip install -e .[dev]
  - pytest
- CI installs ffmpeg on ubuntu via apt in the provided workflow.

License: Apache-2.0