import argparse

from .common import read_config
from .experiment import run
from .report import make_report


def main():
    parser = argparse.ArgumentParser(description="SD 1.4 benign object EEB reproduction")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--stage", choices=["all", "inject", "erase", "generate", "evaluate", "report"], default="all")
    parser.add_argument("--target")
    parser.add_argument("--attack", choices=["none", "shallow", "deep"])
    parser.add_argument("--erasure", choices=["clean", "esd", "uce"])
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.stage != "report":
        rows = run(cfg, args.stage, args.target, args.attack, args.erasure)
        print(f"Completed {args.stage}; evaluated {len(rows)} condition(s).")
    if args.stage in ("all", "report"):
        comparison = make_report(cfg)
        print(f"Wrote comparison table with {len(comparison)} rows.")


if __name__ == "__main__":
    main()
