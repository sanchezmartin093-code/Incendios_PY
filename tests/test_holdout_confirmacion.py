"""Tests for the independent spatial confirmation holdout."""

import unittest
import numpy as np
import polars as pl

from src.holdout import (
    design_confirmation_holdout,
    development_holdout_blocks,
)


class TestHoldoutConfirmacion(unittest.TestCase):
    def test_diseno_determinista_y_disjunto(self):
        xs, ys = [], []
        for i in range(20):
            for j in range(20):
                xs.append(500000.0 + i * 50000.0)
                ys.append(7000000.0 + j * 50000.0)
        grid = pl.DataFrame({
            "cell_id": np.arange(len(xs)),
            "x_utm": xs,
            "y_utm": ys,
        })
        b1, h1, buf1 = design_confirmation_holdout(grid)
        b2, h2, buf2 = design_confirmation_holdout(grid)
        self.assertTrue(b1.equals(b2))
        self.assertTrue(h1.equals(h2))
        self.assertTrue(buf1.equals(buf2))
        self.assertFalse(set(h1["cell_id"]) & set(buf1["cell_id"]))
        dev = development_holdout_blocks(grid)
        conf = set(b1.filter(pl.col("confirmacion"))["block_id"])
        self.assertTrue(dev.isdisjoint(conf))
        self.assertGreater(len(conf), 0)
        self.assertGreater(len(set(h1["cell_id"])), 0)


if __name__ == "__main__":
    unittest.main()
