"""Chinese text auto-blur (cjk_blur.py + core._blur_enable_expr): CJK filter,
box merging, temporal tracking, gap bridging, and the time-ranged ffmpeg
enable expression. Needs no network and no OCR models (reader is stubbed)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from cjk_blur import (
    _detect_frame,
    _detect_pixels,
    _iou,
    _merge_boxes,
    _ocr_one_frame,
    _to_output_box,
    _track_boxes,
    has_cjk,
)
from core import _blur_enable_expr


def _b(x, y, w, h, **kw):
    d = {"x": x, "y": y, "w": w, "h": h}
    d.update(kw)
    return d


class FakeReader:
    """Stub EasyOCR reader: scripted detections; records what it received."""

    def __init__(self, detections):
        self.detections = detections  # [(bbox4, text, conf), ...]
        self.received = []

    def readtext(self, img):
        self.received.append(img)
        return self.detections


def _quad(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


class TestHasCjk(unittest.TestCase):
    def test_chinese(self):
        self.assertTrue(has_cjk("你好世界"))

    def test_mixed(self):
        self.assertTrue(has_cjk("hello 世界"))

    def test_english_only(self):
        self.assertFalse(has_cjk("hello world"))

    def test_empty(self):
        self.assertFalse(has_cjk(""))
        self.assertFalse(has_cjk(None))


class TestMergeBoxes(unittest.TestCase):
    def test_overlapping_merges(self):
        boxes = [_b(0.10, 0.10, 0.10, 0.05), _b(0.15, 0.10, 0.10, 0.05)]
        merged = _merge_boxes(boxes)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0]["x"], 0.10 - 0.008, places=4)

    def test_distant_stays_separate(self):
        boxes = [_b(0.10, 0.10, 0.05, 0.05), _b(0.80, 0.80, 0.05, 0.05)]
        self.assertEqual(len(_merge_boxes(boxes)), 2)

    def test_single_passthrough(self):
        boxes = [_b(0.10, 0.10, 0.05, 0.05)]
        self.assertEqual(len(_merge_boxes(boxes)), 1)


class TestDetectFrame(unittest.TestCase):
    def test_cjk_kept_english_dropped(self):
        reader = FakeReader([
            (_quad(10, 10, 100, 30), "中文字幕", 0.9),
            (_quad(200, 10, 100, 30), "hello", 0.9),
            (_quad(10, 200, 100, 30), "你好", 0.1),  # below min_conf
        ])
        pixels = np.zeros((540, 960, 3), dtype=np.uint8)
        boxes = _detect_pixels(reader, pixels, 960, 540, 0.35)
        # high-conf CJK kept as normal; weak CJK kept flagged (extends only)
        self.assertEqual(len(boxes), 2)
        by_text = {b["text"]: b for b in boxes}
        self.assertFalse(by_text["中文字幕"]["lc"])
        self.assertTrue(by_text["你好"]["lc"])
        # merge pads boxes by 0.008
        self.assertAlmostEqual(by_text["中文字幕"]["x"], 10 / 960 - 0.008, places=4)


class TestTrackBoxes(unittest.TestCase):
    def test_continuous_track(self):
        seq = [(t, [_b(0.2, 0.2, 0.3, 0.1)]) for t in (0.0, 2.0, 4.0, 6.0)]
        tracks = _track_boxes(seq, 2.0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["t0"], 0.0)
        self.assertEqual(tracks[0]["t1"], 6.0)

    def test_gap_bridged(self):
        seq = [
            (0.0, [_b(0.2, 0.2, 0.3, 0.1)]),
            (2.0, []),  # blinked for one sample
            (4.0, [_b(0.2, 0.2, 0.3, 0.1)]),
        ]
        tracks = _track_boxes(seq, 2.0, max_gap_s=8.0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["t1"], 4.0)

    def test_long_gap_stays_split(self):
        seq = [
            (0.0, [_b(0.2, 0.2, 0.3, 0.1)]),
            (2.0, [_b(0.2, 0.2, 0.3, 0.1)]),
            (20.0, [_b(0.2, 0.2, 0.3, 0.1)]),
            (22.0, [_b(0.2, 0.2, 0.3, 0.1)]),
        ]
        tracks = _track_boxes(seq, 2.0, max_gap_s=8.0)
        self.assertEqual(len(tracks), 2)

    def test_interleaved_zones_bridge_correctly(self):
        # Subtitle band blinks; a side name-card appears in between.
        # The two subtitle tracks must still bridge across the name card,
        # and the name card must stay its own track.
        sub = lambda: _b(0.25, 0.81, 0.5, 0.12)   # bottom subtitle band
        card = lambda: _b(0.79, 0.08, 0.12, 0.37)  # right-side name card
        seq = [
            (0.0, [sub()]),
            (2.0, [sub()]),
            (4.0, [card()]),          # name card interleaves
            (6.0, [card()]),
            (10.0, [sub()]),         # subtitle returns after a gap
            (12.0, [sub()]),
        ]
        tracks = _track_boxes(seq, 2.0, max_gap_s=20.0)
        self.assertEqual(len(tracks), 2)
        subs = [t for t in tracks if t["box"]["y"] > 0.5]
        cards = [t for t in tracks if t["box"]["y"] < 0.5]
        self.assertEqual(len(subs), 1)
        self.assertEqual(len(cards), 1)
        self.assertEqual(subs[0]["t0"], 0.0)
        self.assertEqual(subs[0]["t1"], 12.0)
        self.assertEqual(cards[0]["t0"], 4.0)
        self.assertEqual(cards[0]["t1"], 6.0)

    def test_single_frame_dropped_as_noise(self):
        seq = [(0.0, []), (2.0, [_b(0.5, 0.5, 0.2, 0.1)]), (4.0, [])]
        tracks = _track_boxes(seq, 2.0, min_dur_s=1.0)
        self.assertEqual(tracks, [])

    def test_two_zones_tracked_separately(self):
        seq = [(t, [_b(0.1, 0.1, 0.2, 0.1), _b(0.7, 0.7, 0.2, 0.1)])
               for t in (0.0, 2.0, 4.0)]
        tracks = _track_boxes(seq, 2.0)
        self.assertEqual(len(tracks), 2)


class TestOutputBox(unittest.TestCase):
    def test_padding_and_clamp(self):
        tr = {"box": _b(0.0, 0.0, 0.5, 0.5), "t0": 0.5, "t1": 4.0}
        out = _to_output_box(tr, pad_xy=0.015, pad_t=1.0)
        self.assertEqual(out["x"], 0.0)  # clamped, not negative
        self.assertEqual(out["t0"], 0.0)  # clamped, not negative
        self.assertAlmostEqual(out["w"], 0.53, places=3)
        self.assertAlmostEqual(out["t1"], 5.0, places=3)


class TestBlurEnableExpr(unittest.TestCase):
    def test_ranged(self):
        self.assertEqual(
            _blur_enable_expr({"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2,
                               "t0": 1.5, "t1": 4.0}),
            ":enable='between(t,1.500,4.000)'",
        )

    def test_legacy_no_time(self):
        self.assertEqual(
            _blur_enable_expr({"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}), "")

    def test_bad_range(self):
        self.assertEqual(
            _blur_enable_expr({"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2,
                               "t0": 5.0, "t1": 5.0}), "")


class TestIou(unittest.TestCase):
    def test_identical(self):
        b = _b(0.1, 0.1, 0.2, 0.2)
        self.assertAlmostEqual(_iou(b, b), 1.0)

    def test_disjoint(self):
        self.assertEqual(_iou(_b(0.0, 0.0, 0.1, 0.1), _b(0.9, 0.9, 0.1, 0.1)), 0.0)


if __name__ == "__main__":
    unittest.main()

class TestLowConfExtension(unittest.TestCase):
    def test_weak_box_extends_track(self):
        seq = [
            (0.0, [_b(0.2, 0.2, 0.3, 0.1, conf=0.9, lc=False)]),
            (2.0, [_b(0.2, 0.2, 0.3, 0.1, conf=0.9, lc=False)]),
            # weak box (bad recognition) at the same spot extends the track
            (4.0, [_b(0.21, 0.21, 0.28, 0.09, conf=0.08, lc=True)]),
        ]
        tracks = _track_boxes(seq, 2.0, min_dur_s=1.0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["t1"], 4.0)

    def test_weak_box_alone_starts_nothing(self):
        seq = [
            (0.0, []),
            (2.0, [_b(0.5, 0.5, 0.2, 0.1, conf=0.08, lc=True)]),
            (4.0, []),
        ]
        tracks = _track_boxes(seq, 2.0, min_dur_s=1.0)
        self.assertEqual(tracks, [])

    def test_merged_high_plus_weak_can_start_track(self):
        from cjk_blur import _merge_boxes
        merged = _merge_boxes([
            _b(0.20, 0.20, 0.10, 0.05, conf=0.9, lc=False),
            _b(0.22, 0.20, 0.10, 0.05, conf=0.08, lc=True),
        ])
        self.assertEqual(len(merged), 1)
        self.assertFalse(merged[0]["lc"])

class TestLastBoxMatching(unittest.TestCase):
    def test_short_line_matches_previous_short_line(self):
        # union grows wide from an early long line; later short lines must
        # still match via the most recent box, not the wide union.
        seq = [
            (0.0, [_b(0.20, 0.80, 0.60, 0.10, conf=0.9)]),   # long line
            (2.0, [_b(0.40, 0.80, 0.10, 0.10, conf=0.9)]),   # short line
            (4.0, [_b(0.41, 0.80, 0.10, 0.10, conf=0.9)]),   # short line again
        ]
        tracks = _track_boxes(seq, 2.0, min_dur_s=1.0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["t1"], 4.0)

class TestStrongShapeWeakBoxes(unittest.TestCase):
    def _detect(self, w, h, text="陈实", conf=0.02):
        # fake easyocr readtext -> (bbox, text, conf); frame 960x540
        px = lambda v, s: v * s
        box = [[px(0.10, 960), px(0.10, 540)],
               [px(0.10 + w, 960), px(0.10, 540)],
               [px(0.10 + w, 960), px(0.10 + h, 540)],
               [px(0.10, 960), px(0.10 + h, 540)]]
        class R:
            def readtext(self, p):
                return [(box, text, conf)]
        return _detect_pixels(R(), np.zeros((540, 960, 3), np.uint8), 960, 540, 0.35)

    def test_vertical_weak_box_can_start_track(self):
        boxes = self._detect(0.04, 0.40)  # tall-narrow: vertical text
        self.assertEqual(len(boxes), 1)
        self.assertFalse(boxes[0]["lc"])
        tracks = _track_boxes([(0.0, boxes), (2.0, boxes)], 2.0, min_dur_s=1.0)
        self.assertEqual(len(tracks), 1)

    def test_large_weak_box_can_start_track(self):
        boxes = self._detect(0.45, 0.20, text="本剧为")  # large: disclaimer
        self.assertEqual(len(boxes), 1)
        self.assertFalse(boxes[0]["lc"])

    def test_small_weak_box_stays_extend_only(self):
        boxes = self._detect(0.10, 0.05)  # small, weak -> extend-only
        self.assertEqual(len(boxes), 1)
        self.assertTrue(boxes[0]["lc"])

class TestSingleFrameStrongTrack(unittest.TestCase):
    def test_brief_vertical_card_survives(self):
        # a name card visible on a single sampled frame must not be dropped
        b = {"x": 0.81, "y": 0.09, "w": 0.12, "h": 0.38, "text": "陈夏花",
             "conf": 0.02, "lc": False, "ss": True}
        tracks = _track_boxes([(44.0, [b])], 2.0, min_dur_s=1.0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["t0"], 44.0)

    def test_single_weak_normal_box_still_dropped(self):
        b = {"x": 0.30, "y": 0.80, "w": 0.10, "h": 0.05, "text": "你好",
             "conf": 0.40, "lc": False, "ss": False}
        tracks = _track_boxes([(44.0, [b])], 2.0, min_dur_s=1.0)
        self.assertEqual(tracks, [])

class TestStrongShapeNoCjk(unittest.TestCase):
    def _detect(self, w, h, text="...", conf=0.0):
        box = [[96, 54], [96 + w * 960, 54],
               [96 + w * 960, 54 + h * 540], [96, 54 + h * 540]]
        class R:
            def readtext(self, p):
                return [(box, text, conf)]
        return _detect_pixels(R(), np.zeros((540, 960, 3), np.uint8), 960, 540, 0.35)

    def test_vertical_no_cjk_kept_as_strong(self):
        # stylized vertical text: recognition fails entirely, shape saves it
        boxes = self._detect(0.04, 0.40, text="", conf=0.0)
        self.assertEqual(len(boxes), 1)
        self.assertFalse(boxes[0]["lc"])
        self.assertTrue(boxes[0]["ss"])

    def test_small_english_still_dropped(self):
        boxes = self._detect(0.10, 0.05, text="hello", conf=0.9)
        self.assertEqual(len(boxes), 0)

class TestEdgeTouchRule(unittest.TestCase):
    def _detect_at(self, x0n, y0n, w, h, text="", conf=0.0):
        box = [[x0n * 960, y0n * 540], [(x0n + w) * 960, y0n * 540],
               [(x0n + w) * 960, (y0n + h) * 540], [x0n * 960, (y0n + h) * 540]]
        class R:
            def readtext(self, p):
                return [(box, text, conf)]
        return _detect_pixels(R(), np.zeros((540, 960, 3), np.uint8), 960, 540, 0.35)

    def test_top_edge_no_cjk_dropped(self):
        # tree branches touching the top edge: not trusted without text
        boxes = self._detect_at(0.05, 0.000, 0.075, 0.30)
        self.assertEqual(len(boxes), 0)

    def test_padded_no_cjk_kept(self):
        # same shape with padding from the edge: trusted as vertical text
        boxes = self._detect_at(0.07, 0.12, 0.04, 0.40)
        self.assertEqual(len(boxes), 1)
        self.assertTrue(boxes[0]["ss"])


class TestNonAsciiPath(unittest.TestCase):
    """Regression: Windows + Chinese path must not kill the blur pass.

    EasyOCR's internal cv2.imread() returns None for non-ASCII paths on
    Windows (e.g. a job folder named after a Chinese video), which crashed
    with "'NoneType' object has no attribute 'shape'" and skipped the
    entire auto-blur pass. The reader must receive pixels, never a path.
    """

    def test_reader_never_gets_a_path(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "辰雪奔赴1-8超长版_acc6af97" / "frm_00001.jpg"
            img_path.parent.mkdir(parents=True)
            Image.new("RGB", (96, 54), (20, 20, 20)).save(img_path)
            reader = FakeReader([(_quad(10, 10, 40, 12), "中文字", 0.9)])
            boxes = _detect_frame(reader, img_path, 96, 54, 0.35)
            self.assertEqual(len(reader.received), 1)
            got = reader.received[0]
            self.assertIsInstance(got, np.ndarray,
                                  "EasyOCR must get pixels, not a file path")
            self.assertEqual(got.shape, (54, 96, 3))
            self.assertEqual(len(boxes), 1)
            self.assertEqual(boxes[0]["text"], "中文字")

    def test_one_bad_frame_does_not_abort(self):
        class BoomReader:
            def readtext(self, img):
                raise RuntimeError("simulated OCR failure")

        logs: list[str] = []
        boxes = _ocr_one_frame(BoomReader(), Path("nope.jpg"), 0.35,
                               logs.append, 2, 10)
        self.assertEqual(boxes, [])
        self.assertTrue(any("frame 2/10" in m for m in logs),
                        f"expected a frame-tagged warning, got: {logs}")
