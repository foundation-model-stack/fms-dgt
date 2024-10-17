# Standard
import os
import shutil

# Third Party
import yaml

# Local
from fms_dgt.utils import load_yaml_config


def test_include_yaml():

    dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    os.makedirs(dir_path)
    os.makedirs(os.path.join(dir_path, "layer"))

    dict1 = {
        "a": "b",
        "c": "d",
        "include": ["layer/../././././la*er/fil*e2.yaml", {"g": ["./layer/*"]}],
    }
    dict2 = {"e": "f"}
    with (
        open(os.path.join(dir_path, "file1.yaml"), "w") as f1,
        open(os.path.join(dir_path, "layer/file2.yaml"), "w") as f2,
    ):
        yaml.dump(dict1, f1)
        yaml.dump(dict2, f2)

    res = load_yaml_config(os.path.join(dir_path, "file1.yaml"))

    assert "e" in res and res["e"] == "f", f"Error with include directive"

    shutil.rmtree(dir_path)
