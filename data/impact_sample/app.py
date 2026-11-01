"""Entry point."""
from shop.checkout import checkout


def main():
    """Run a demo checkout."""
    return checkout([10, 20], 10)


if __name__ == "__main__":
    main()