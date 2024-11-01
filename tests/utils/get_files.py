from icevision.utils.get_files import get_image_files


def test_get_image_files():
    fns = get_image_files("../../samples/images")
    assert len(fns) == 6
