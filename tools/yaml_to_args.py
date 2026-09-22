import sys

import yaml


def main(config_path):
    config = yaml.safe_load(open(config_path)) or {}
    for key, value in config.items():
        flag = f"--{key.replace('_', '-')}"
        if value is True:
            print(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, list):
            print(flag, *map(str, value), sep="\n")
        else:
            print(flag, str(value), sep="\n")


if __name__ == "__main__":
    main(sys.argv[1])
