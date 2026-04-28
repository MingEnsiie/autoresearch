import ast
from pathlib import Path
import unittest


def _literal_assignments(path: Path):
    tree = ast.parse(path.read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Pow):
                    values[node.targets[0].id] = ast.literal_eval(node.value.left) ** ast.literal_eval(node.value.right)
    return values


class TrainConfigTests(unittest.TestCase):
    def test_default_device_batch_size_is_single_gpu_friendly(self):
        values = _literal_assignments(Path(__file__).resolve().parents[1] / "train.py")
        self.assertLessEqual(values["DEVICE_BATCH_SIZE"], 32)

    def test_total_batch_size_stays_divisible_by_fwd_bwd_tokens(self):
        values = _literal_assignments(Path(__file__).resolve().parents[1] / "train.py")
        tokens_per_fwdbwd = values["DEVICE_BATCH_SIZE"] * 2048
        self.assertEqual(values["TOTAL_BATCH_SIZE"] % tokens_per_fwdbwd, 0)


if __name__ == "__main__":
    unittest.main()
