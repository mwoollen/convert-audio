"""Tests for transcription module."""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from convert_audio.transcribe import format_timestamp


def test_format_timestamp():
    """Test timestamp formatting."""
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(1.5) == "00:00:01,500"
    assert format_timestamp(65.123) == "00:01:05,123"
    assert format_timestamp(3665.789) == "01:01:05,789"


def test_transcribe_audio_whisper_not_installed(monkeypatch):
    """Test that missing whisper dependency is handled gracefully."""
    from convert_audio.transcribe import transcribe_audio

    # Mock whisper import to fail
    def mock_import(name, *args, **kwargs):
        if name == "whisper":
            raise ImportError("No module named 'whisper'")
        return __import__(name, *args, **kwargs)

    with patch('builtins.__import__', side_effect=mock_import):
        result = transcribe_audio("test.mp3", backend="whisper")
        assert result == 1


def test_transcribe_audio_faster_whisper_not_installed(monkeypatch):
    """Test that missing faster-whisper dependency is handled gracefully."""
    from convert_audio.transcribe import transcribe_audio

    # Mock faster_whisper import to fail
    def mock_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("No module named 'faster_whisper'")
        return __import__(name, *args, **kwargs)

    with patch('builtins.__import__', side_effect=mock_import):
        result = transcribe_audio("test.mp3", backend="faster-whisper")
        assert result == 1


def test_transcribe_audio_basic(tmp_path):
    """Test basic transcription functionality."""
    from convert_audio.transcribe import transcribe_audio

    # Setup mocks
    mock_model = MagicMock()
    mock_whisper = MagicMock()
    mock_whisper.load_model.return_value = mock_model
    mock_model.transcribe.return_value = {
        "text": "Hello world",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello world"}
        ]
    }

    # Create test input file
    input_file = tmp_path / "test.mp3"
    input_file.write_text("fake audio")

    # Patch whisper at import time
    with patch.dict('sys.modules', {'whisper': mock_whisper}):
        # Run transcription with whisper backend
        result = transcribe_audio(str(input_file), model="base", backend="whisper")

        # Verify
        assert result == 0
        mock_whisper.load_model.assert_called_once_with("base")
        mock_model.transcribe.assert_called_once_with(
            str(input_file),
            language=None,
            verbose=False,
        )

        # Check output file was created
        output_file = tmp_path / "test.txt"
        assert output_file.exists()
        assert output_file.read_text() == "Hello world"


def test_transcribe_audio_with_language(tmp_path):
    """Test transcription with language specified."""
    from convert_audio.transcribe import transcribe_audio

    mock_model = MagicMock()
    mock_whisper = MagicMock()
    mock_whisper.load_model.return_value = mock_model
    mock_model.transcribe.return_value = {
        "text": "Hola mundo",
        "segments": []
    }

    input_file = tmp_path / "test.mp3"
    input_file.write_text("fake audio")

    with patch.dict('sys.modules', {'whisper': mock_whisper}):
        result = transcribe_audio(str(input_file), language="es", backend="whisper")

        assert result == 0
        mock_model.transcribe.assert_called_once_with(
            str(input_file),
            language="es",
            verbose=False,
        )


def test_transcribe_audio_srt_format(tmp_path):
    """Test SRT subtitle generation."""
    from convert_audio.transcribe import transcribe_audio

    mock_model = MagicMock()
    mock_whisper = MagicMock()
    mock_whisper.load_model.return_value = mock_model
    mock_model.transcribe.return_value = {
        "text": "Hello world",
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "Hello"},
            {"start": 1.5, "end": 3.0, "text": "world"}
        ]
    }

    input_file = tmp_path / "test.mp3"
    input_file.write_text("fake audio")
    output_file = tmp_path / "test.srt"

    with patch.dict('sys.modules', {'whisper': mock_whisper}):
        result = transcribe_audio(
            str(input_file),
            output_format="srt",
            output_path=str(output_file),
            backend="whisper"
        )

        assert result == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert "00:00:00,000 --> 00:00:01,500" in content
        assert "Hello" in content
        assert "world" in content


def test_transcribe_audio_faster_whisper_basic(tmp_path):
    """Test basic transcription functionality with faster-whisper."""
    from convert_audio.transcribe import transcribe_audio

    # Setup mocks for faster-whisper
    mock_model = MagicMock()
    mock_whisper_model_class = MagicMock(return_value=mock_model)

    # Mock the transcribe method to return generator
    mock_segment1 = MagicMock()
    mock_segment1.start = 0.0
    mock_segment1.end = 1.0
    mock_segment1.text = " Hello world"

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.95

    mock_model.transcribe.return_value = (iter([mock_segment1]), mock_info)

    # Create test input file
    input_file = tmp_path / "test.mp3"
    input_file.write_text("fake audio")

    # Mock faster_whisper module
    mock_faster_whisper = MagicMock()
    mock_faster_whisper.WhisperModel = mock_whisper_model_class

    with patch.dict('sys.modules', {'faster_whisper': mock_faster_whisper}):
        # Run transcription with faster-whisper backend
        result = transcribe_audio(str(input_file), model="base", backend="faster-whisper")

        # Verify
        assert result == 0
        mock_whisper_model_class.assert_called_once_with("base", device="cpu", compute_type="int8")

        # Check output file was created
        output_file = tmp_path / "test.txt"
        assert output_file.exists()
        assert output_file.read_text().strip() == "Hello world"


def test_transcribe_audio_whisperx_not_installed(monkeypatch):
    """Test that missing whisperx dependency is handled gracefully."""
    from convert_audio.transcribe import transcribe_audio

    # Mock whisperx import to fail
    def mock_import(name, *args, **kwargs):
        if name == "whisperx":
            raise ImportError("No module named 'whisperx'")
        return __import__(name, *args, **kwargs)

    with patch('builtins.__import__', side_effect=mock_import):
        result = transcribe_audio("test.mp3", backend="whisperx")
        assert result == 1


def test_transcribe_audio_whisperx_basic(tmp_path):
    """Test basic transcription functionality with whisperx."""
    from convert_audio.transcribe import transcribe_audio

    # Setup mocks for whisperx
    mock_whisperx = MagicMock()
    mock_model = MagicMock()
    mock_audio = MagicMock()

    mock_whisperx.load_model.return_value = mock_model
    mock_whisperx.load_audio.return_value = mock_audio

    mock_model.transcribe.return_value = {
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": " Hello world"}
        ]
    }

    # Create test input file
    input_file = tmp_path / "test.mp3"
    input_file.write_text("fake audio")

    # Mock whisperx and torch modules
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch.dict('sys.modules', {'whisperx': mock_whisperx, 'torch': mock_torch}):
        # Run transcription with whisperx backend
        result = transcribe_audio(str(input_file), model="base", backend="whisperx")

        # Verify
        assert result == 0
        mock_whisperx.load_model.assert_called_once()
        mock_whisperx.load_audio.assert_called_once_with(str(input_file))
        mock_model.transcribe.assert_called_once()

        # Check output file was created
        output_file = tmp_path / "test.txt"
        assert output_file.exists()
        assert output_file.read_text().strip() == "Hello world"
