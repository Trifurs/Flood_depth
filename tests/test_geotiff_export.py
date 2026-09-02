from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin

from utils.raster_io import write_geotiff


def test_atomic_geotiff_round_trip(tmp_path) -> None:
    values = np.arange(35, dtype=np.float32).reshape(5, 7)
    valid = np.ones_like(values, dtype=bool)
    valid[0, 0] = False
    transform = from_origin(12345.0, 54321.0, 20.0, 20.0)
    output = write_geotiff(
        tmp_path / "depth.tif",
        values,
        crs="EPSG:27704",
        transform=transform,
        nodata=-9999.0,
        valid_mask=valid,
        descriptions=["predicted_depth_m"],
    )
    with rasterio.open(output) as dataset:
        observed = dataset.read(1)
        assert dataset.crs.to_string() == "EPSG:27704"
        assert dataset.transform == transform
        assert dataset.nodata == -9999.0
        assert (dataset.height, dataset.width) == values.shape
        assert dataset.descriptions == ("predicted_depth_m",)
        assert observed[0, 0] == -9999.0
        np.testing.assert_allclose(observed[valid], values[valid])
