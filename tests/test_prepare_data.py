from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
rgb_from_hdf5 = pytest.importorskip("prepare_data").rgb_from_hdf5


def test_nyu_hdf5_axis_conversion(tmp_path: Path):
    path = tmp_path / "tiny.mat"
    source = np.arange(1 * 3 * 4 * 2, dtype=np.uint8).reshape(1, 3, 4, 2)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("images", data=source)
    with h5py.File(path, "r") as handle:
        converted = rgb_from_hdf5(handle["images"], 0)
    assert converted.shape == (2, 4, 3)
    assert np.array_equal(converted, source[0].transpose(2, 1, 0))
