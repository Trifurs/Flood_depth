from __future__ import annotations

from pathlib import Path

import rasterio

from tools.smoke_test import run_smoke


def test_real_smoke_training_checkpoint_reload_and_export(config_path: Path, tmp_path) -> None:
    report = run_smoke(
        config_path,
        device_name="auto",
        train_batches=2,
        val_batches=1,
        output_root=tmp_path,
    )
    assert report["status"] == "passed"
    assert Path(report["checkpoint"]).is_file()
    prediction = Path(report["prediction_geotiff"])
    assert prediction.is_file()
    with rasterio.open(prediction) as dataset:
        assert dataset.crs.to_string() == "EPSG:27704"
        assert dataset.width == dataset.height == 256
        assert dataset.nodata == -9999.0
