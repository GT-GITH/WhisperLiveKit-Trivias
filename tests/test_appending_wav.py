"""Test voor _AppendingWavWriter (hervatten na een onderbreking, plan fase 3).

Draai: `python tests/test_appending_wav.py`. Laadt alleen de klasse uit
audio_processor.py (zonder de zware imports van die module).
"""

import ast
import struct
import tempfile
import wave
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "whisperlivekit" / "audio_processor.py"
_tree = ast.parse(_SRC.read_text(encoding="utf-8"))
_cls = next(n for n in _tree.body if isinstance(n, ast.ClassDef) and n.name == "_AppendingWavWriter")
_ns: dict = {"Path": Path}
exec(compile(ast.Module(body=[_cls], type_ignores=[]), str(_SRC), "exec"), _ns)
AppendingWavWriter = _ns["_AppendingWavWriter"]


def _make_wav(path, frames: bytes):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(frames)


def test_append_keeps_old_audio_and_header_valid():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.wav"
        first = struct.pack("<4h", 1, 2, 3, 4)
        _make_wav(p, first)
        w = AppendingWavWriter(p)
        w.writeframes(struct.pack("<2h", 5, 6))
        # kop moet al geldig zijn vóór close (batch-worker leest tijdens de opname)
        with wave.open(str(p), "rb") as r:
            assert r.getnframes() == 6
        w.writeframes(struct.pack("<h", 7))
        w.close()
        with wave.open(str(p), "rb") as r:
            assert r.getnframes() == 7
            assert struct.unpack("<7h", r.readframes(7)) == (1, 2, 3, 4, 5, 6, 7)


def test_truncates_garbage_after_data():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.wav"
        _make_wav(p, struct.pack("<2h", 1, 2))
        with open(p, "ab") as f:
            f.write(b"\x00\x01\x02")  # rommel na een crash
        w = AppendingWavWriter(p)
        w.writeframes(struct.pack("<h", 3))
        w.close()
        with wave.open(str(p), "rb") as r:
            assert struct.unpack("<3h", r.readframes(r.getnframes())) == (1, 2, 3)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"{len(tests)} tests geslaagd")
