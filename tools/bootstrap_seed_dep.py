import sys


def main():
    if "--enable" not in sys.argv:
        print(
            "Dependency-based seed bootstrap is disabled. Run with --enable after adding domain patterns.",
            file=sys.stderr,
        )
        sys.exit(2)
    # Placeholder: add spaCy Matcher/DependencyMatcher logic here in the future.
    print("Dependency-based bootstrap not implemented yet.")
    sys.exit(2)


if __name__ == "__main__":
    main()



