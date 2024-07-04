import ee
from ee_kfold import sets

ee.Initialize(project="ncs-gc")


def test_sets():
    img = ee.Image().constant(1)
    region = ee.Geometry.Rectangle([-112, 34, -111, 35])
    training_data = img.sample(region, numPixels=100)
    s = sets(training_data).getInfo()
    assert isinstance(s, list)

def test_kfold
