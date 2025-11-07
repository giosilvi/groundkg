import sys


def main():
    if "--enable" not in sys.argv:
        print(
            "LLM-backed seed bootstrap is disabled. Run with --enable after configuring provider and gating.",
            file=sys.stderr,
        )
        print(
            "Suggested gating: short E1..E2 distance, compatible types, high-confidence prompts; audit samples.",
            file=sys.stderr,
        )
        sys.exit(2)
    # Placeholder: integrate an LLM client and guarded sampling here.
    print("LLM bootstrap not implemented yet.")
    sys.exit(2)


if __name__ == "__main__":
    main()



