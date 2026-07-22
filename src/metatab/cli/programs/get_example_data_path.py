"""
Program that returns to stdout the example dataset path.
It is meant to work with command substitution to get the example dataset filepath.
"""
from metatab.metatab_utils.package_data import get_example_data_path

def main():
    print(get_example_data_path())

if __name__ == "__main__":
    main()