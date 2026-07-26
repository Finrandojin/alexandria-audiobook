"""Regenerate or verify the checked-in unittest inventory."""

import argparse
import json
from pathlib import Path

from test_inventory import (EXCLUDED_TEST_MODULES, INVENTORY_PATH,
                            _tracked_test_modules, get_unit_test_inventory)


def format_inventory(inventory):
    """Return the deterministic on-disk representation of an inventory."""
    return json.dumps(inventory, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def get_inventory_differences(expected, actual):
    """Return compact, actionable differences between two inventories."""
    differences = []
    for module_name in sorted(set(expected) | set(actual)):
        expected_tests = set(expected.get(module_name, []))
        actual_tests = set(actual.get(module_name, []))
        for test_id in sorted(actual_tests - expected_tests):
            differences.append(f"Missing from inventory: {test_id}")
        for test_id in sorted(expected_tests - actual_tests):
            differences.append(f"No longer discovered: {test_id}")
    return differences


def check_inventory(path=INVENTORY_PATH):
    """Compare the checked-in inventory with current test discovery."""
    expected = json.loads(Path(path).read_text(encoding="utf-8"))
    return get_inventory_differences(expected, get_unit_test_inventory())


def write_inventory(path=INVENTORY_PATH):
    """Write the current inventory deterministically and return its path."""
    path = Path(path)
    path.write_text(format_inventory(get_unit_test_inventory()), encoding="utf-8")
    return path


def find_untracked_test_modules():
    """Test files on disk that git does not track, so are not inventoried.

    The inventory deliberately covers only tracked files, because this tree is
    shared with other branches. The cost is an ordering trap: write a new test,
    regenerate, commit, and CI discovers tests the inventory never saw. Warning
    here turns a red build into a line of output.
    """
    tracked = _tracked_test_modules()
    if tracked is None:
        return []
    return sorted(
        path.name for path in Path(__file__).parent.glob("test*.py")
        if path.stem not in EXCLUDED_TEST_MODULES and path.stem not in tracked)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="report drift without modifying the inventory")
    args = parser.parse_args(argv)
    if args.check:
        differences = check_inventory()
        if differences:
            print("Unit test inventory drift detected:")
            for difference in differences:
                print(f"- {difference}")
            return 1
        print("Unit test inventory matches discovery.")
        return 0
    path = write_inventory()
    print(f"Updated {path}")
    untracked = find_untracked_test_modules()
    if untracked:
        print("\nWARNING: these test files are untracked and are NOT in the "
              "inventory.\nIf they belong to this change, 'git add' them and "
              "run this again, or CI\nwill discover tests the inventory does "
              "not list:")
        for name in untracked:
            print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
