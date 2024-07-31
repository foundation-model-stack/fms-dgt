# Third Party
import pytest

# Local
from fms_dgt.utils import sanitize_path


def test_sanitize_path():
    assert sanitize_path("../test") == "test"
    assert sanitize_path("../../test") == "test"
    assert sanitize_path("../../abc/../test") == "test"
    assert sanitize_path("../../abc/../test/fixtures") == "test/fixtures"
    assert sanitize_path("../../abc/../.test/fixtures") == ".test/fixtures"
    assert sanitize_path("/test/foo") == "test/foo"
    assert sanitize_path("./test/bar") == "test/bar"
    assert sanitize_path(".test/baz") == ".test/baz"
    assert sanitize_path("qux") == "qux"
