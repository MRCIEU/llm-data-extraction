from yiutils.project_utils import find_project_root


def main():
    proj_root = find_project_root("justfile")
    print(f"Project root: {proj_root}")


if __name__ == "__main__":
    main()
