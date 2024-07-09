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
    k = 10
    model = ee.Classifier.smileRandomForest(k)
    results = run_kfold(model, training_data, classProperty="y", inputProperties=["X"])

    assert len(results) == k

    some_keys = list(results[0].keys())
    some_keys.sort()

    assert some_keys == [
        "classified_test_set",
        "classified_training_set",
        "model",
    ]

    mean_accuracy = fmean(
        [
            r["classified_test_set"]
            .errorMatrix("y", "classification")
            .accuracy()
            .getInfo()
            for r in results
        ]
    )
    assert isinstance(mean_accuracy, float)
