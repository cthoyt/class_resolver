# -*- coding: utf-8 -*-

"""Tests for the class resolver."""

import itertools
import unittest

import click
from click.testing import CliRunner, Result

from class_resolver import Resolver, UnexpectedKeywordError

try:
    import ray.tune as tune
except ImportError:
    tune = None


class Base:
    """A base class."""

    def __init__(self, name):
        """Initialize the class."""
        self.name = name

    def __eq__(self, other) -> bool:
        """Check two instances are equal."""
        return type(self) == type(other) and self.name == other.name


class A(Base):
    """A base class."""


class B(Base):
    """B base class."""


class C(Base):
    """C base class."""


class D(Base):
    """D base class."""


class AltBase:
    """An alternative base class."""


class AAltBase(AltBase):
    """A base class."""


class TestResolver(unittest.TestCase):
    """Tests for the resolver."""

    def setUp(self) -> None:
        """Set up the resolver class."""
        self.resolver = Resolver([A, B, C], base=Base)

    def test_contents(self):
        """Test the functions."""
        self.assertIn(A, set(self.resolver))

    def test_iterator(self):
        """Test iterating over classes."""
        self.assertEqual([A, B, C], list(self.resolver))

    def test_lookup(self):
        """Test looking up classes."""
        self.assertEqual(A, self.resolver.lookup("a"))
        self.assertEqual(A, self.resolver.lookup("A"))
        with self.assertRaises(ValueError):
            self.resolver.lookup(None)
        with self.assertRaises(KeyError):
            self.resolver.lookup("missing")
        with self.assertRaises(TypeError):
            self.resolver.lookup(3)

    def test_passthrough(self):
        """Test instances are passed through unmodified."""
        a = A(name="charlie")
        self.assertEqual(a, self.resolver.make(a))

    def test_make(self):
        """Test making classes."""
        name = "charlie"
        # Test instantiating with positional dict into kwargs
        self.assertEqual(A(name=name), self.resolver.make("a", {"name": name}))
        # Test instantiating with kwargs
        self.assertEqual(A(name=name), self.resolver.make("a", name=name))

    def test_registration_synonym(self):
        """Test failure of registration."""
        self.resolver.register(D, synonyms={"dope"})
        name = "charlie"
        self.assertEqual(D(name=name), self.resolver.make("d", name=name))

    def test_registration_failure(self):
        """Test failure of registration."""
        with self.assertRaises(KeyError):
            self.resolver.register(A)
        with self.assertRaises(KeyError):
            self.resolver.register(D, synonyms={"a"})

    def test_make_from_kwargs(self):
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
    def test_variant_generation(self):
        """Test whether ray tune can generate variants from the search space."""
        search_space = self.resolver.ray_tune_search_space(
            kwargs_search_space=dict(
                name=tune.choice(["charlie", "max"]),
            ),
        )
        for spec in itertools.islice(
            tune.suggest.variant_generator.generate_variants(search_space), 2
        ):
            config = {k[0]: v for k, v in spec[0].items()}
            query = config.pop("query")
            instance = self.resolver.make(query=query, pos_kwargs=config)
            self.assertIsInstance(instance, Base)

    def test_bad_click_option(self):
        """Test failure to get a click option."""
        with self.assertRaises(ValueError):
            self.resolver.get_option("--opt")  # no default given

    def test_click_option(self):
        """Test the click option."""

        @click.command()
        @self.resolver.get_option("--opt", default="a")
        def cli(opt):
            """Run the test CLI."""
            self.assertIsInstance(opt, type)
            click.echo(opt.__name__, nl=False)

        self._test_cli(cli)

    def _test_cli(self, cli):
        runner = CliRunner()

        # Test default
        result: Result = runner.invoke(cli, [])
        self.assertEqual(A.__name__, result.output)

        # Test canonical name
        result: Result = runner.invoke(cli, ["--opt", "A"])
        self.assertEqual(A.__name__, result.output)

        # Test normalizing name
        result: Result = runner.invoke(cli, ["--opt", "a"])
        self.assertEqual(A.__name__, result.output)

    def test_click_option_str(self):
        """Test the click option."""

        @click.command()
        @self.resolver.get_option("--opt", default="a", as_string=True)
        def cli(opt):
            """Run the test CLI."""
            self.assertIsInstance(opt, str)
            click.echo(self.resolver.lookup(opt).__name__, nl=False)

        self._test_cli(cli)

    def test_signature(self):
        """Check signature tests."""
        self.assertTrue(self.resolver.supports_argument("A", "name"))
        self.assertFalse(self.resolver.supports_argument("A", "nope"))

    def test_no_arguments(self):
        """Check that the unexpected keyword error is thrown properly."""
        resolver = Resolver.from_subclasses(AltBase)
        with self.assertRaises(UnexpectedKeywordError) as e:
            resolver.make("A", nope="nopppeeee")
            self.assertEqual("AAltBase did not expect any keyword arguments", str(e))
