"""Run the synthetic PLC as a separate loopback service."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--root", type=Path, default=Path("runs/mock"))
    parser.add_argument("--manual-clock", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    import uvicorn
    from copper_mvp.control.mock_api import create_mock_app
    uvicorn.run(create_mock_app(args.root, manual_clock=args.manual_clock, seed=args.seed),
                host="127.0.0.1", port=args.port, access_log=False)
