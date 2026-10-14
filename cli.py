"""BijliPredict AI - unified command-line interface.

Wraps the individual scripts behind a single entry point (installed as `bijli`):

    bijli data [options]       # generate the synthetic dataset
    bijli train [options]      # train the GRU / LSTM forecaster
    bijli evaluate [options]   # evaluate on the held-out test split
    bijli predict [options]    # 7-day forecast + 90-day projection + bill
    bijli report [options]     # build the Excel report from outputs/

Defaults come from config.toml; any explicit flag overrides them. The original
scripts keep working unchanged (`python train.py ...`).

Examples:
    bijli train --epochs 20
    bijli predict --preset bses
    bijli --config other.toml train
"""

from __future__ import annotations

import argparse
import importlib
import sys
import tomllib
from pathlib import Path

DEFAULT_CONFIG = "config.toml"

# command -> (module name, one-line help). The module must expose build_parser()
# and main(args). Modules are imported lazily so `bijli data` need not load torch.
COMMANDS: dict[str, tuple[str, str]] = {
    "data": ("generate_data", "generate the synthetic household dataset"),
    "train": ("train", "train the GRU / LSTM forecaster"),
    "evaluate": ("evaluate", "evaluate the model on the held-out test split"),
    "predict": ("predict", "forecast consumption, cost and the bill"),
    "report": ("report", "build the Excel report from outputs/"),
}


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Load a TOML config file. A missing file yields empty defaults."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _split_global(argv: list[str]) -> tuple[str, list[str]]:
    """Pull the global `--config` (allowed anywhere) out of argv."""
    pre = argparse.ArgumentParser(prog="bijli", add_help=False)
    pre.add_argument("--config", default=DEFAULT_CONFIG)
    ns, rest = pre.parse_known_args(argv)
    return ns.config, rest


def build_parser(command: str, module, config: dict) -> argparse.ArgumentParser:
    """Build the subcommand parser with config.toml values as defaults."""
    parser = module.build_parser()
    parser.prog = f"bijli {command}"
    known = {action.dest for action in parser._actions}
    section = config.get(command, {})
    parser.set_defaults(**{k: v for k, v in section.items() if k in known})
    return parser


def _print_help() -> None:
    print("usage: bijli [--config PATH] <command> [options]\n")
    print("commands:")
    for name, (_, help_text) in COMMANDS.items():
        print(f"  {name:<9} {help_text}")
    print("\nRun 'bijli <command> --help' for command options.")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path, rest = _split_global(argv)

    if not rest or rest[0] in ("-h", "--help"):
        _print_help()
        return

    command, cmd_args = rest[0], rest[1:]
    if command not in COMMANDS:
        choices = ", ".join(COMMANDS)
        raise SystemExit(f"bijli: unknown command '{command}' (choose from: {choices})")

    module_name, _ = COMMANDS[command]
    module = importlib.import_module(module_name)
    config = load_config(config_path)
    parser = build_parser(command, module, config)
    args = parser.parse_args(cmd_args)
    args.config = config_path
    module.main(args)


if __name__ == "__main__":
    main()
