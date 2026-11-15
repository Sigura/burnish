import sys
import traceback


def main(namespace):
    tests = [(name, fn) for name, fn in namespace.items() if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failed += 1
            tb = traceback.format_exc().strip().splitlines()
            print(f"FAIL {name}: " + "\n  ".join(tb[-4:]))
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
