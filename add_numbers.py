"""
Add two numbers. Use from command line or import in Python.

Command line:
  python add_numbers.py 3 5
  → 8

In code:
  from add_numbers import add
  add(3, 5)  # 8
"""


def add(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python add_numbers.py <number1> <number2>")
        sys.exit(1)
    try:
        x = float(sys.argv[1])
        y = float(sys.argv[2])
        result = add(x, y)
        print(result)
    except ValueError:
        print("Error: both arguments must be numbers")
        sys.exit(1)
