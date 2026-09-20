# tests/test_pipeline.py
import pytest
from src.image_pipeline import generate_image

def test_pipeline_import():
    assert callable(generate_image)