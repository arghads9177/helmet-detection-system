"""Integration tests for scripts/preprocessing/extract_frames.py and clean_frames.py.

Uses synthetic frames/video generated in-test -- no committed binary fixtures needed for
these pipeline-mechanics checks.
"""
import cv2
import numpy as np
import pytest

from scripts.preprocessing.clean_frames import clean_frames, is_blank, is_blurry
from scripts.preprocessing.extract_frames import extract_frames, sampling_stride
from PIL import Image, ImageFilter


def _make_video(path, fps=30, n_frames=90, size=(64, 48)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), fill_value=(i * 2) % 256, dtype=np.uint8)
        frame[:5, :5] = rng.integers(0, 256, size=(5, 5, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestSamplingStride:
    def test_matches_target_fps(self):
        assert sampling_stride(source_fps=30.0, target_fps=1.5) == 20

    def test_never_below_one(self):
        assert sampling_stride(source_fps=10.0, target_fps=30.0) == 1

    def test_rejects_non_positive_target(self):
        with pytest.raises(ValueError):
            sampling_stride(source_fps=30.0, target_fps=0)


class TestExtractFrames:
    def test_writes_expected_frame_count(self, tmp_path):
        video_path = tmp_path / "clip.mp4"
        _make_video(video_path, fps=30, n_frames=90)
        out_dir = tmp_path / "frames"

        written = extract_frames(video_path, out_dir, target_fps=1.5)

        assert written == 5  # stride 20 over 90 frames -> indices 0, 20, 40, 60, 80
        assert len(list(out_dir.glob("*.jpg"))) == written

    def test_raises_on_missing_video(self, tmp_path):
        with pytest.raises(OSError):
            extract_frames(tmp_path / "missing.mp4", tmp_path / "out", target_fps=1.5)


class TestCleanFrames:
    def test_drops_blank_frame(self):
        blank = Image.new("RGB", (32, 32), color=(120, 120, 120))
        assert is_blank(blank) is True

    def test_keeps_textured_frame(self):
        rng = np.random.default_rng(1)
        arr = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        textured = Image.fromarray(arr)
        assert is_blank(textured) is False

    def test_drops_blurry_frame(self):
        checkerboard = np.indices((64, 64)).sum(axis=0) % 2 * 255
        sharp = Image.fromarray(checkerboard.astype(np.uint8)).convert("RGB")
        blurred = sharp.filter(ImageFilter.GaussianBlur(radius=8))
        assert is_blurry(sharp) is False
        assert is_blurry(blurred) is True

    def test_dedups_near_identical_frames_and_drops_blank(self, tmp_path):
        source_dir = tmp_path / "src"
        source_dir.mkdir()

        rng = np.random.default_rng(2)
        base = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        Image.fromarray(base).save(source_dir / "frame_000.jpg")
        # Near-duplicate: tiny perturbation, should be deduped away.
        near_dup = base.copy()
        near_dup[0, 0] = (near_dup[0, 0].astype(np.int16) + 1) % 256
        Image.fromarray(near_dup).save(source_dir / "frame_001.jpg")
        # Distinct frame: should survive.
        distinct = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        Image.fromarray(distinct).save(source_dir / "frame_002.jpg")
        # Blank frame: should be dropped.
        Image.new("RGB", (32, 32), color=(10, 10, 10)).save(source_dir / "frame_003.jpg")

        out_dir = tmp_path / "cleaned"
        kept, total = clean_frames(source_dir, out_dir)

        assert total == 4
        assert kept == 2
        assert (out_dir / "frame_000.jpg").exists()
        assert (out_dir / "frame_002.jpg").exists()
        assert not (out_dir / "frame_001.jpg").exists()
        assert not (out_dir / "frame_003.jpg").exists()

    def test_drops_corrupt_file(self, tmp_path):
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        (source_dir / "corrupt.jpg").write_bytes(b"not a real image")

        out_dir = tmp_path / "cleaned"
        kept, total = clean_frames(source_dir, out_dir)

        assert total == 1
        assert kept == 0
