#!/usr/bin/env python3
"""
Standalone Agent Orchestrator — CLI Entry Point.

Usage:
    python3 standalone_main.py "Create a hello world function"
    python3 standalone_main.py --resume
    python3 standalone_main.py "Build a REST API" --max-iterations 5
"""

import sys
import argparse
import logging
from pathlib import Path

from standalone_config import load_config
from standalone_orchestrator import Orchestrator


def setup_logging(verbose: bool = False, log_file: Path = None):
    """Configure logging with colors."""
    level = logging.DEBUG if verbose else logging.INFO

    COLORS = {
        'DEBUG': '\033[36m', 'INFO': '\033[32m', 'WARNING': '\033[33m',
        'ERROR': '\033[31m', 'CRITICAL': '\033[35m', 'RESET': '\033[0m',
    }

    class ColorFormatter(logging.Formatter):
        def format(self, record):
            color = COLORS.get(record.levelname, '')
            reset = COLORS['RESET']
            record.levelname = f"{color}{record.levelname:<8}{reset}"
            return super().format(record)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(ColorFormatter(
        '%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s',
        datefmt='%H:%M:%S'
    ))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s'
        ))
        root.addHandler(fh)

    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Agent Orchestrator — Local Multi-Agent Execution System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Create a hello world function"
  %(prog)s "Build a REST API" --max-iterations 5
  %(prog)s --resume
  %(prog)s "Fix the login bug" --config my_config.json -v
        """
    )
    parser.add_argument(
        "task", nargs="?",
        help="The task/goal to accomplish"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing session state"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help="Maximum iterations before escalation (default: from config)"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Configuration JSON file path"
    )
    parser.add_argument(
        "--working-dir", type=Path, default=Path.cwd(),
        help="Working directory for the task (default: current directory)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose/debug logging"
    )
    parser.add_argument(
        "--log-file", type=Path, default=None,
        help="Write logs to file"
    )

    args = parser.parse_args()

    # Validate args
    if not args.task and not args.resume:
        parser.error("Must provide a task or use --resume")

    # Setup
    setup_logging(verbose=args.verbose, log_file=args.log_file)

    config = load_config(args.config)
    if args.max_iterations is not None:
        config.max_iterations = args.max_iterations

    working_dir = args.working_dir.resolve()

    # Run
    orchestrator = Orchestrator(config, working_dir)

    try:
        success = orchestrator.run(args.task or "", resume=args.resume)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("\n⚠️ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.getLogger(__name__).exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
