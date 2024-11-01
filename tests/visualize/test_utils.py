import pytest

from icevision.visualize.utils import as_rgb_tuple


@pytest.mark.parametrize(
    "color_input",
    [
        ("red"),
        ("#fff"),
        ("#ffaaaa"),
        ([255, 255, 255]),
        ((255, 255, 255)),
    ],
)
def test_as_rgb_tuple(color_input):
    assert as_rgb_tuple(color_input)
