# Standard
from dataclasses import dataclass
from typing import Any

# Local
from fms_dgt.blocks.utilities.field_map import FieldMapBlock


@dataclass
class TestDataClass:
    field1: Any = None
    field2: Any = None
    field3: Any = None
    field4: Any = None


def test_field_map_single():
    block = FieldMapBlock(
        name="test_field_map",
        field_map={"field1": "field2", "field3": "field4", "field2": "field3"},
    )
    test_data = [TestDataClass(field1=1, field2=2, field3=3, field4=4)]
    block(test_data)
    assert (
        test_data[0].field1 == 1
        and test_data[0].field2 == 1
        and test_data[0].field3 == 2
        and test_data[0].field4 == 3
    )


def test_field_map_multi():
    block = FieldMapBlock(name="test_field_map", field_map={"field1": "field2"})
    test_data = [TestDataClass(field1=1, field2=2, field3=3, field4=4)]
    block(test_data)
    block = FieldMapBlock(name="test_field_map", field_map={"field2": "field3"})
    block(test_data)
    assert (
        test_data[0].field1 == 1
        and test_data[0].field2 == 1
        and test_data[0].field3 == 1
    )


def test_field_map_dict():
    block = FieldMapBlock(
        name="test_field_map",
        field_map={"field1": "field2", "field3": "field4", "field2": "field3"},
    )
    test_data = [{"field1": 1, "field2": 2, "field3": 3, "field4": 4}]
    block(test_data)
    assert (
        test_data[0]["field1"] == 1
        and test_data[0]["field2"] == 1
        and test_data[0]["field3"] == 2
        and test_data[0]["field4"] == 3
    )
