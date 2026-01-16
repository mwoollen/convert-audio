"""Tests for CLI module."""

import pytest
from convert_audio.cli import main


def test_cli_requires_input(monkeypatch):
    """Test that CLI requires input argument."""
    monkeypatch.setattr("sys.argv", ["convert-audio"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2  # argparse error code


def test_cli_help(monkeypatch, capsys):
    """Test that --help works."""
    monkeypatch.setattr("sys.argv", ["convert-audio", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Convert and transcribe audio files" in captured.out
