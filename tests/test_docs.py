"""Docs must match the code they describe."""

from scripts.gen_config_docs import OUTPUT, render
from src.features.engineer import FeatureEngineer
from tests.factories import make_ohlcv

DOCS = OUTPUT.parent


def test_configuration_page_is_up_to_date():
    assert OUTPUT.read_text(encoding='utf-8') == render(), (
        "docs/configuration.md is out of date; run: python -m scripts.gen_config_docs"
    )


def test_guides_list_every_feature():
    engineer = FeatureEngineer()
    engineer.compute_all_features(make_ohlcv(250))
    guides = (DOCS / 'guides.md').read_text(encoding='utf-8')

    missing = [name for name in engineer.feature_names if f'`{name}`' not in guides]
    assert not missing, f"Add these features to the table in docs/guides.md: {missing}"
