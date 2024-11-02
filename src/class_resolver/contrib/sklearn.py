"""
Scikit-learn is a generic machine learning package with implementations of
algorithms for classification, regression, dimensionality reduction, clustering,
as well as other generic tooling.

The ``class-resolver`` provides several class resolvers for instantiating various
implementations, such as those of linear models.
"""  # noqa: D205

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    RidgeClassifierCV,
    SGDClassifier,
)
from sklearn.tree import DecisionTreeClassifier

from ..api import ClassResolver

__all__ = [
    "classifier_resolver",
]

classifier_resolver: ClassResolver[BaseEstimator] = ClassResolver(
    [
        LogisticRegression,
        LogisticRegressionCV,
        PassiveAggressiveClassifier,
        Perceptron,
        RidgeClassifier,
        RidgeClassifierCV,
        SGDClassifier,
        DecisionTreeClassifier,
        RandomForestClassifier,
        GradientBoostingClassifier,
    ],
    base=BaseEstimator,
    base_as_suffix=False,
    default=LogisticRegression,
    synonyms={"lr": LogisticRegression, "logreg": LogisticRegression, "sgd": SGDClassifier},
    location="class_resolver.contrib.sklearn.classifier_resolver",
)
"""A resolver for classifiers.

The default value is :class:`sklearn.linear_model.LogisticRegression`.
This resolver can be used like in the following:

.. code-block:: python

    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    from class_resolver.contrib.sklearn import classifier_resolver

    # Prepare a dataset
    x, y = datasets.load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # Lookup with a string
    classifier = classifier_resolver.make("LogisticRegression")
    classifier.fit(x_train, y_train)
    assert 0.7 < classifier.score(x_test, y_test)

    # Default lookup gives logistic regression
    classifier = classifier_resolver.make(None)
    classifier.fit(x_train, y_train)
    assert 0.7 < classifier.score(x_test, y_test)

.. seealso:: https://scikit-learn.org/stable/modules/classes.html#linear-classifiers
"""
