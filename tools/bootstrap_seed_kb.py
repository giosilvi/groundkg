import sys


def main():
    if "--kb" not in " ".join(sys.argv):
        print(
            "KB-backed seed bootstrap is disabled. Provide --kb <path> after integrating a KB.",
            file=sys.stderr,
        )
        sys.exit(2)
    # Placeholder: implement KB-driven distant supervision here.
    print("KB bootstrap not implemented yet.")
    sys.exit(2)


if __name__ == "__main__":
    main()



