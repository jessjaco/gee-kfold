from statistics import fmean

import ee
from ee_kfold import sets, run_kfold
import pytest

ee.Initialize(project="ncs-gc")


@pytest.fixture
def training_data() -> ee.FeatureCollection:
    slope = ee.Terrain.slope(ee.Image("USGS/3DEP/10m"))
    y = slope.lt([5, 10, 20]).reduce("sum").rename("y")
    X = slope.add(ee.Image.random().multiply(15)).rename("X")
    region = ee.Geometry.Rectangle([-112, 34, -111, 35])
    return X.addBands(y).sample(region, numPixels=100)


def test_sets(training_data):
    s = sets(training_data)
    assert isinstance(s, list)


def test_kfold(training_data):
    model = ee.Classifier.smileRandomForest(10)
    results = run_kfold(model, training_data, classProperty="y", inputProperties=["X"])

    mean_accuracy = fmean(
        [r[2].errorMatrix("y", "classification").accuracy().getInfo() for r in results]
    )
    print(mean_accuracy)
    breakpoint()
