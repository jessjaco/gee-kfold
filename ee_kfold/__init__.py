import ee


def sets(training_data: ee.FeatureCollection, k: int = 10, seed: int | float = 1337):
    test_proportion = 1.0 / k
    training_data = training_data.randomColumn("r", seed)

    return [
        (
            training_data.filter(
                ee.Filter.Or(
                    ee.Filter.lt("r", i * test_proportion),
                    ee.Filter.gte("r", (i + 1) * test_proportion),
                )
            ),
            training_data.filter(
                ee.Filter.Or(
                    ee.Filter.gte("r", i * test_proportion),
                    ee.Filter.lt("r", (i + 1) * test_proportion),
                )
            ),
        )
        for i in range(k)
    ]


def run_kfold(model: ee.Classifier, training_data: ee.FeatureCollection, **kwargs):
    results = []
    for train, test in sets(training_data):
        trained_model = model.train(features=train, **kwargs)
        train_result = train.classify(trained_model)
        test_result = test.classify(trained_model)
        results.append(
            dict(
                model=trained_model,
                classified_training_set=train_result,
                classified_test_set=test_result,
            )
        )

    return results
