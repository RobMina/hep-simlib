"""python -m hepsimlib.capture audit|account|bank ... (see the package docstring)."""
import argparse
import json
import sys
from pathlib import Path


def audit_main(argv):
    from .audit import audit
    from .tables import read, read_links, require
    p = argparse.ArgumentParser(prog="python -m hepsimlib.capture audit", description="strict capture audit")
    p.add_argument("input", type=Path)
    p.add_argument("--directory", default="capture")
    p.add_argument("--expected-events", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    require(not args.output.exists(), "output already exists; refusing to clobber")
    result = audit(read(args.input, args.directory), args.expected_events,
                   links=read_links(args.input, args.directory))
    result["input"] = args.input.name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("audit", "account", "bank"):
        print(__doc__.strip(), file=sys.stderr)
        return 2
    command, rest = argv[0], argv[1:]
    if command == "audit":
        audit_main(rest)
    elif command == "account":
        from .accounting import main as account
        account(rest)
    else:
        from .bank import main as bank
        bank(rest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
