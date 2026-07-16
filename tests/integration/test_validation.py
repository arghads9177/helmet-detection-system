"""Integration tests for scripts/validation/check_annotations.py and check_dataset.py.

Uses synthetic fixtures generated in-test -- no committed binary fixtures needed for these
pipeline-mechanics checks.
"""
import numpy as np
from PIL import Image

from scripts.validation.check_annotations import check_label_file
from scripts.validation.check_dataset import find_corrupt, find_near_duplicates, find_orphans


class TestCheckAnnotations:
    def test_accepts_valid_boxes(self, tmp_path):
        label_path = tmp_path / "img.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.05 0.05\n")

        report = check_label_file(label_path, num_classes=2)

        assert report.ok
        assert report.box_count == 2

    def test_flags_malformed_line(self, tmp_path):
        label_path = tmp_path / "img.txt"
        label_path.write_text("0 0.5 0.5 0.2\n")  # missing a token

        report = check_label_file(label_path, num_classes=2)

        assert not report.ok
        assert "malformed" in report.issues[0]

    def test_flags_out_of_range_class_id(self, tmp_path):
        label_path = tmp_path / "img.txt"
        label_path.write_text("5 0.5 0.5 0.2 0.2\n")

        report = check_label_file(label_path, num_classes=2)

        assert not report.ok
        assert "out of range" in report.issues[0]

    def test_flags_out_of_bounds_box(self, tmp_path):
        label_path = tmp_path / "img.txt"
        label_path.write_text("0 0.95 0.5 0.5 0.2\n")  # extends past x=1

        report = check_label_file(label_path, num_classes=2)

        assert not report.ok
        assert "outside image bounds" in report.issues[0]

    def test_flags_non_positive_dimensions(self, tmp_path):
        label_path = tmp_path / "img.txt"
        label_path.write_text("0 0.5 0.5 0.0 0.2\n")

        report = check_label_file(label_path, num_classes=2)

        assert not report.ok
        assert "non-positive" in report.issues[0]

    def test_flags_duplicate_box(self, tmp_path):
        label_path = tmp_path / "img.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n")

        report = check_label_file(label_path, num_classes=2)

        assert not report.ok
        assert "duplicate" in report.issues[0]
        assert report.box_count == 1


class TestCheckDataset:
    def test_finds_corrupt_file(self, tmp_path):
        good = tmp_path / "good.jpg"
        Image.new("RGB", (32, 32), color=(1, 2, 3)).save(good)
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not an image")

        corrupt = find_corrupt([good, bad])

        assert corrupt == [bad]

    def test_finds_near_duplicates(self, tmp_path):
        rng = np.random.default_rng(0)
        base = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        a = tmp_path / "a.jpg"
        Image.fromarray(base).save(a)
        b = tmp_path / "b.jpg"
        Image.fromarray(base).save(b)  # identical -> near-duplicate
        distinct = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        c = tmp_path / "c.jpg"
        Image.fromarray(distinct).save(c)

        duplicates = find_near_duplicates([a, b, c])

        assert (a, b) in duplicates
        assert not any(c in pair for pair in duplicates)

    def test_finds_orphans(self, tmp_path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        annotations_dir = tmp_path / "annotations"
        annotations_dir.mkdir()

        img_with_label = images_dir / "img1.jpg"
        Image.new("RGB", (8, 8)).save(img_with_label)
        img_without_label = images_dir / "img2.jpg"
        Image.new("RGB", (8, 8)).save(img_without_label)
        (annotations_dir / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (annotations_dir / "img3.txt").write_text("0 0.5 0.5 0.2 0.2\n")  # orphan label

        orphan_images, orphan_labels = find_orphans([img_with_label, img_without_label], annotations_dir)

        assert orphan_images == [img_without_label]
        assert orphan_labels == [annotations_dir / "img3.txt"]
