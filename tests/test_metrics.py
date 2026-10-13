import unittest

from evaluation import metrics as M


class TestMetrics(unittest.TestCase):
    def test_basic(self):
        yt, yp = [1, 1, 0, 0], [1, 0, 0, 1]
        self.assertEqual(M.accuracy(yt, yp), 0.5)
        self.assertEqual(M.precision_recall_f1(yt, yp), (0.5, 0.5, 0.5))

    def test_auc(self):
        self.assertEqual(M.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(M.roc_auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertIsNone(M.roc_auc([1, 1], [0.3, 0.4]))  # single class -> no value

    def test_empty_returns_none(self):
        self.assertIsNone(M.accuracy([], []))
        self.assertEqual(M.precision_recall_f1([0, 0], [0, 0]), (None, None, None))

    def test_tiou_and_matching(self):
        self.assertAlmostEqual(M.temporal_iou((0, 10), (5, 15)), 5 / 15)
        self.assertEqual(M.temporal_iou((0, 1), (2, 3)), 0.0)
        m, fp, fn = M.match_events([(0, 10), (50, 60)], [(1, 9)], 0.5)
        self.assertEqual((len(m), fp, fn), (1, 1, 0))

    def test_macro_f1_mae_confusion(self):
        self.assertEqual(M.mae([1, 3], [2, 3]), 0.5)
        self.assertEqual(M.confusion_matrix([1, 2], [1, 1], [1, 2]).tolist(), [[1, 0], [1, 0]])
        self.assertAlmostEqual(M.macro_f1([1, 2], [1, 2], [1, 2]), 1.0)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            M.accuracy([1], [1, 0])


if __name__ == "__main__":
    unittest.main()
