"""Tests for the class resolver."""

import itertools
import unittest
from collections.abc import Collection, Sequence
from typing import Any, ClassVar, Optional, cast

import click
from click.testing import CliRunner, Result
from docdata import parse_docdata

from class_resolver import (
    VERSION,
    ClassResolver,
    KeywordArgumentError,
    RegistrationNameConflict,
    RegistrationSynonymConflict,
    Resolver,
    UnexpectedKeywordError,
)

try:
    import ray.tune as tune
except ImportError:
    tune = None

try:
    import optuna
except ImportError:
    optuna = None

try:
    import sklearn
except ImportError:
    sklearn = None


class Base:
    """A base class."""

    def __init__(self, name: str) -> None:
        """Initialize the class."""
        self.name = name

    def __eq__(self, other: Any) -> bool:
        """Check two instances are equal."""
        return type(self) is type(other) and self.name == other.name


@parse_docdata
class A(Base):
    """A base class.

    ---
    k1: v1
    k2:
        k21: v21
    """

    synonyms: ClassVar[Collection[str]] = {"a_synonym_1", "a_synonym_2"}


class B(Base):
    """B base class."""


class C(Base):
    """C base class."""


class D(Base):
    """D base class."""


class E(Base):
    """E base class."""

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize the class."""
        super().__init__(name or "default_name")


class AltBase:
    """An alternative base class."""


class AAltBase(AltBase):
    """A base class."""


class TestResolver(unittest.TestCase):
    """Tests for the resolver."""

    def setUp(self) -> None:
        """Set up the resolver class."""
        self.resolver = Resolver([A, B, C, E], base=Base)

    def test_version(self) -> None:
        """Test version."""
        self.assertIsInstance(VERSION, str)

    def test_contents(self) -> None:
        """Test the functions."""
        self.assertIn(A, set(self.resolver))

    def test_iterator(self) -> None:
        """Test iterating over classes."""
        self.assertEqual([A, B, C, E], list(self.resolver))

    def test_normalize(self) -> None:
        """Test normalize functions."""
        self.assertEqual("e", self.resolver.normalize_cls(E))
        self.assertEqual("e", self.resolver.normalize_cls(E))

        instance = E()
        self.assertEqual("e", self.resolver.normalize_inst(instance))

    def test_lookup(self) -> None:
        """Test looking up classes."""
        self.assertEqual(A, self.resolver.lookup("a"))
        self.assertEqual(A, self.resolver.lookup("A"))
        self.assertEqual(A, self.resolver.lookup("a_synonym_1"))
        self.assertEqual(A, self.resolver.lookup("a_synonym_2"))
        self.assertEqual(A, self.resolver.lookup(None, default=A))
        with self.assertRaises(ValueError):
            self.resolver.lookup(None)
        with self.assertRaises(KeyError):
            self.resolver.lookup("missing")
        with self.assertRaises(TypeError):
            self.resolver.lookup(3)  # type:ignore
        with self.assertRaises(TypeError) as e:
            self.resolver.lookup(AAltBase)  # type:ignore
        self.assertEqual(f"Not subclass of {self.resolver.base.__name__}: {AAltBase}", str(e.exception))
        self.assertEqual(self.resolver.lookup(A(name="max")), A)

    def test_docdata(self) -> None:
        """Test docdata."""
        full = {
            "k1": "v1",
            "k2": {"k21": "v21"},
        }
        self.assertEqual(full, self.resolver.docdata("a"))
        self.assertEqual("v1", self.resolver.docdata("a", "k1"))
        self.assertEqual({"k21": "v21"}, self.resolver.docdata("a", "k2"))
        self.assertEqual("v21", self.resolver.docdata("a", "k2", "k21"))

    def test_lookup_no_synonyms(self) -> None:
        """Test looking up classes without auto-synonym."""
        resolver = Resolver([A], base=Base, synonym_attribute=None)
        self.assertEqual(A, resolver.lookup("a"))
        self.assertEqual(A, resolver.lookup("A"))
        with self.assertRaises(KeyError):
            self.assertEqual(A, resolver.lookup("a_synonym_1"))

    def test_passthrough(self) -> None:
        """Test instances are passed through unmodified."""
        a = A(name="charlie")
        self.assertEqual(a, self.resolver.make(a))

    def test_make(self) -> None:
        """Test making classes."""
        name = "charlie"
        # Test instantiting with positional dict into kwargs
        self.assertEqual(A(name=name), self.resolver.make("a", {"name": name}))
        # Test instantiating with kwargs
        self.assertEqual(A(name=name), self.resolver.make("a", name=name))

    def test_make_safe(self) -> None:
        """Test the make_safe function, which always returns none on none input."""
        self.assertIsNone(self.resolver.make_safe(None))
        self.assertIsNone(Resolver.from_subclasses(Base, default=A).make_safe(None))

        name = "charlie"
        # Test instantiating with positional dict into kwargs
        self.assertEqual(A(name=name), self.resolver.make_safe("a", {"name": name}))
        # Test instantiating with kwargs
        self.assertEqual(A(name=name), self.resolver.make_safe("a", name=name))

    def test_registration_synonym(self) -> None:
        """Test failure of registration."""
        self.assertNotIn(D, self.resolver.lookup_dict.values())
        self.resolver.register(D, synonyms={"dope"})
        name = "charlie"
        self.assertEqual(D(name=name), self.resolver.make("d", name=name))

    def test_registration_empty_synonym_failure(self) -> None:
        """Test failure of registration."""
        self.assertNotIn(D, self.resolver.lookup_dict.values())
        with self.assertRaises(ValueError):
            self.resolver.register(D, synonyms={""})

    def test_registration_name_failure(self) -> None:
        """Test failure of registration."""
        with self.assertRaises(RegistrationNameConflict) as e:
            self.resolver.register(A)
        self.assertEqual("name", e.exception.label)
        self.assertIn("name", str(e.exception))
        with self.assertRaises(RegistrationNameConflict) as e:
            self.resolver.register(D, synonyms={"a"})
        self.assertEqual("synonym", e.exception.label)
        self.assertIn("synonym", str(e.exception))

    def test_registration_synonym_failure(self) -> None:
        """Test failure of registration."""
        resolver = Resolver([], base=Base)
        resolver.register(A, synonyms={"B"})
        with self.assertRaises(RegistrationSynonymConflict) as e:
            resolver.register(B)
        self.assertEqual("name", e.exception.label)
        self.assertIn("name", str(e.exception))

        class F(Base):
            """Extra class for testing."""

        with self.assertRaises(RegistrationSynonymConflict) as e:
            resolver.register(F, synonyms={"B"})
        self.assertEqual("synonym", e.exception.label)
        self.assertIn("synonym", str(e.exception))

    def test_make_from_kwargs(self) -> None:
        """Test making classes from kwargs."""
        name = "charlie"
        self.assertEqual(
            A(name=name),
            self.resolver.make_from_kwargs(
                key="magic",
                data=dict(
                    ignored_entry=...,
                    magic="a",
                    magic_kwargs=dict(
                        name=name,
                    ),
                ),
            ),
        )

    @unittest.skipIf(tune is None, "ray[tune] was not installed properly")
    @unittest.skipIf(
        tune is not None and getattr(tune, "suggest", None) is None,
        "a newer version of ray[tune] is installed that does not have the tune.suggest module",
    )
    def test_variant_generation(self) -> None:
        """Test whether ray tune can generate variants from the search space."""
        search_space = self.resolver.ray_tune_search_space(
            kwargs_search_space=dict(
                name=tune.choice(["charlie", "max"]),
            ),
        )
        for spec in itertools.islice(tune.suggest.variant_generator.generate_variants(search_space), 2):
            config = {k[0]: v for k, v in spec[0].items()}
            query = config.pop("query")
            instance = self.resolver.make(query=query, pos_kwargs=config)
            self.assertIsInstance(instance, Base)

    @unittest.skipIf(optuna is None, "optuna is not installed")
    @unittest.skipIf(sklearn is None, "sklearn is not installed")
    def test_optuna_suggest(self) -> None:
        """Test suggesting categorical for optuna."""
        import optuna
        from sklearn import datasets
        from sklearn.base import BaseEstimator
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        resolver = ClassResolver(
            [
                LogisticRegression,
                RandomForestClassifier,
            ],
            base=BaseEstimator,
            base_as_suffix=False,
            default=LogisticRegression,
        )

        def objective(trial: optuna.Trial) -> float:
            """Calculate the classification accuracy for the iris dataset."""
            x, y = datasets.load_iris(return_X_y=True)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
            clf_cls = resolver.optuna_lookup(trial, "model")
            clf = clf_cls()
            clf.fit(x_train, y_train)
            return cast(float, clf.score(x_test, y_test))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)

    def test_bad_click_option(self) -> None:
        """Test failure to get a click option."""
        with self.assertRaises(ValueError):
            self.resolver.get_option("--opt")  # no default given

    def test_required_click_option(self) -> None:
        """Test non-failure to get a required click option without default."""
        self.resolver.get_option("--opt", as_string=True, required=True)

    def test_click_option(self) -> None:
        """Test the click option."""

        @click.command()  # type:ignore
        @self.resolver.get_option("--opt", default="a")
        def cli(opt) -> None:
            """Run the test CLI."""
            self.assertIsInstance(opt, type)
            click.echo(opt.__name__, nl=False)

        self._test_cli(cli)

    def _test_cli(self, cli: click.Command) -> None:
        runner = CliRunner()

        # Test default
        result_1: Result = runner.invoke(cli, [])
        self.assertEqual(A.__name__, result_1.output)

        # Test canonical name
        result_2: Result = runner.invoke(cli, ["--opt", "A"])
        self.assertEqual(A.__name__, result_2.output)

        # Test normalizing name
        result_3: Result = runner.invoke(cli, ["--opt", "a"])
        self.assertEqual(A.__name__, result_3.output)

        # Test synonym lookup works correctly
        result_4: Result = runner.invoke(cli, ["--opt", "asynonym1"])
        self.assertEqual(A.__name__, result_4.output)

    def test_click_option_str(self) -> None:
        """Test the click option."""

        @click.command()  # type:ignore
        @self.resolver.get_option("--opt", default="a", as_string=True)
        def cli(opt: str):
            """Run the test CLI."""
            self.assertIsInstance(opt, str)
            click.echo(self.resolver.lookup(opt).__name__, nl=False)

        self._test_cli(cli)

    def test_click_option_default(self) -> None:
        """Test generating an option with a default."""
        resolver = Resolver([A, B, C, E], base=Base, default=A)

        @click.command()  # type:ignore
        @resolver.get_option("--opt", as_string=True)
        def cli(opt: str) -> None:
            """Run the test CLI."""
            self.assertIsInstance(opt, str)
            click.echo(self.resolver.lookup(opt).__name__, nl=False)

        self._test_cli(cli)

    def test_click_option_multiple(self) -> None:
        """Test the click option with multiple arguments."""

        @click.command()  # type:ignore
        @self.resolver.get_option("--opt", default="a", as_string=True, multiple=True)
        def cli(opt: Sequence[str]) -> None:
            """Run the test CLI."""
            self.assertIsInstance(opt, Sequence)
            for opt_ in opt:
                self.assertIsInstance(opt_, str)
                click.echo(self.resolver.lookup(opt_).__name__, nl=False)

        self._test_cli(cli)

    def test_signature(self) -> None:
        """Check signature tests."""
        self.assertTrue(self.resolver.supports_argument("A", "name"))
        self.assertFalse(self.resolver.supports_argument("A", "nope"))

    def test_no_arguments(self) -> None:
        """Check that the unexpected keyword error is thrown properly."""
        resolver = Resolver.from_subclasses(AltBase)
        with self.assertRaises(UnexpectedKeywordError) as e:
            resolver.make("A", nope="nopppeeee")
        self.assertEqual("AAltBase did not expect any keyword arguments", str(e.exception))

    def test_base_suffix(self) -> None:
        """Check that the unexpected keyword error is thrown properly."""
        resolver = Resolver.from_subclasses(AltBase, suffix=None, base_as_suffix=True)
        self.assertEqual(AAltBase, resolver.lookup("AAltBase"))
        self.assertEqual(AAltBase, resolver.lookup("A"))

        resolver = Resolver.from_subclasses(AltBase, suffix="nope", base_as_suffix=True)
        self.assertEqual(AAltBase, resolver.lookup("AAltBase"))
        with self.assertRaises(KeyError):
            resolver.lookup("A")

        resolver = Resolver.from_subclasses(AltBase, suffix="")
        self.assertEqual(AAltBase, resolver.lookup("AAltBase"))
        with self.assertRaises(KeyError):
            resolver.lookup("A")

        resolver = Resolver.from_subclasses(AltBase, base_as_suffix=False)
        self.assertEqual(AAltBase, resolver.lookup("AAltBase"))
        with self.assertRaises(KeyError):
            resolver.lookup("A")

    def test_make_many(self) -> None:
        """Test the make_many function."""
        with self.assertRaises(ValueError):
            # no default is given
            self.resolver.make_many(None)

        with self.assertRaises(ValueError):
            # wrong number of kwargs is given
            self.resolver.make_many([], [{}, {}])

        with self.assertRaises(ValueError):
            # wrong number of kwargs is given
            self.resolver.make_many(["a", "a", "a"], [{}, {}])

        # One class, one kwarg
        instances = self.resolver.make_many("a", dict(name="name"))
        self.assertEqual([A(name="name")], instances)
        instances = self.resolver.make_many("a", [dict(name="name")])
        self.assertEqual([A(name="name")], instances)
        instances = self.resolver.make_many(["a"], dict(name="name"))
        self.assertEqual([A(name="name")], instances)
        instances = self.resolver.make_many(["a"], [dict(name="name")])
        self.assertEqual([A(name="name")], instances)

        # Single class, multiple kwargs
        instances = self.resolver.make_many("a", [dict(name="name1"), dict(name="name2")])
        self.assertEqual([A(name="name1"), A(name="name2")], instances)
        instances = self.resolver.make_many(["a"], [dict(name="name1"), dict(name="name2")])
        self.assertEqual([A(name="name1"), A(name="name2")], instances)

        # Multiple class, one kwargs
        instances = self.resolver.make_many(["a", "b", "c"], dict(name="name"))
        self.assertEqual([A(name="name"), B(name="name"), C(name="name")], instances)
        instances = self.resolver.make_many(["a", "b", "c"], [dict(name="name")])
        self.assertEqual([A(name="name"), B(name="name"), C(name="name")], instances)

        # Multiple class, multiple kwargs
        instances = self.resolver.make_many(
            ["a", "b", "c"], [dict(name="name1"), dict(name="name2"), dict(name="name3")]
        )
        self.assertEqual([A(name="name1"), B(name="name2"), C(name="name3")], instances)

        # One class, No kwargs
        instances = self.resolver.make_many("e")
        self.assertEqual([E()], instances)
        instances = self.resolver.make_many(["e"])
        self.assertEqual([E()], instances)
        instances = self.resolver.make_many("e", None)
        self.assertEqual([E()], instances)
        instances = self.resolver.make_many(["e"], None)
        self.assertEqual([E()], instances)
        instances = self.resolver.make_many(["e"], [None])
        self.assertEqual([E()], instances)

        # No class
        resolver = Resolver.from_subclasses(Base, default=A)
        instances = resolver.make_many(None, dict(name="name"))
        self.assertEqual([A(name="name")], instances)

    def test_missing_kwarg(self) -> None:
        """Test error on missing kwarg."""

        class Alt2Base:
            """Another alternative base class."""

            def __init__(self, *, kwarg: Any) -> None:
                """Initialize the class."""
                self.kwarg = kwarg

        class AAlt2Base(Alt2Base):
            """Another base class."""

        resolver = Resolver.from_subclasses(Alt2Base)
        with self.assertRaises(KeywordArgumentError) as e:
            resolver.make("a")
        self.assertIn("required keyword-only", str(e.exception))

    def test_unexpected_error(self) -> None:
        """Test an arbitrary type error thrown during making a class."""

        class Alt3Base:
            """Another alternative base class."""

            def __init__(self) -> None:
                """Initialize the class."""
                raise TypeError("surprise!")

        class AAlt3Base(Alt3Base):
            """Another base class."""

        resolver = Resolver.from_subclasses(Alt3Base)
        with self.assertRaises(TypeError) as e:
            resolver.make("a")
        self.assertEqual("surprise!", str(e.exception))
