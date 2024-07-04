import ee


def sets(
    training_data: ee.FeatureCollection, k: int = 10, seed: int | float = 1337
) -> ee.List:
    test_proportion = ee.Number(1.0 / k)
    training_data = training_data.randomColumn("r", seed)

    return ee.List.sequence(
        0, ee.Number(1).subtract(test_proportion), test_proportion
    ).map(
        lambda lower_limit: (
            training_data.filter(
                ee.Filter.Or(
                    ee.Filter.lt("r", lower_limit),
                    ee.Filter.gte("r", ee.Number(lower_limit).add(test_proportion)),
                )
            ),
            training_data.filter(
                ee.Filter.Or(
                    ee.Filter.gte("r", lower_limit),
                    ee.Filter.lt("r", ee.Number(lower_limit).add(test_proportion)),
                )
            ),
        )
    )


def run_kfold(model: ee.Classifier, training_data: ee.FeatureCollection, **kwargs):
    results = []
    for train, test in sets(training_data, **kwargs):
        trained_model = model.train(features=train)
        results.append(test.classify(trained_model))

    return results
