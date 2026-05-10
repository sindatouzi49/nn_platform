"""Launch the Neural Networks Platform.

    python run.py            # http://127.0.0.1:5000
    python run.py --port 8000
"""
import argparse
from backend.app import app

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
