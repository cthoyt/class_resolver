"""Indeed."""

import click
from sklearn.base import BaseEstimator

from class_resolver.contrib.sklearn import classifier_resolver


@click.command()
@classifier_resolver.get_option("--classifier")
def main(classifier: type[BaseEstimator]):
    """Do it."""
    click.echo(f"You picked: {classifier}")


if __name__ == "__main__":
    main()
