"""
Pytest fixtures for dental_agent test suite.
"""

import pytest
from PIL import Image
import pandas as pd

from dental_agent.tools.synthetic import make_synthetic_dental_image


@pytest.fixture
def synthetic_image() -> Image.Image:
    """Fixture providing a synthetic 1024x512 dental panoramic X-ray."""
    return make_synthetic_dental_image(width=1024, height=512, seed=42)


@pytest.fixture
def sample_annotations_df() -> pd.DataFrame:
    """Fixture providing sample DENTEX annotations DataFrame."""
    return pd.DataFrame([
        {
            "id": 1,
            "image_id": 100,
            "category_id_1": 1,  # Maxillary Right
            "category_id_2": 6,  # First Molar (FDI 16)
            "category_id_3": 1,  # Caries
            "bbox": [500, 200, 80, 100],
        },
        {
            "id": 2,
            "image_id": 100,
            "category_id_1": 3,  # Mandibular Left
            "category_id_2": 8,  # Wisdom tooth (FDI 38)
            "category_id_3": 4,  # Impacted tooth
            "bbox": [800, 350, 90, 110],
        },
    ])


@pytest.fixture
def sample_categories_df() -> pd.DataFrame:
    """Fixture providing sample DENTEX category lookup DataFrame."""
    return pd.DataFrame([
        {"id": 1, "name": "Caries"},
        {"id": 2, "name": "Deep Caries"},
        {"id": 3, "name": "Periapical Lesion"},
        {"id": 4, "name": "Impacted Tooth"},
    ])
