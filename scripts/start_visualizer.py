#!/usr/bin/env python3
"""
Phase 3: Start Visualizer Script

Starts the web UI server for exploring the Atlas.

All parameters read from config.json under "visualizer" section.

Input:
    - Documents from SQLite
    - Mappings from Parquet/JSON (optional)
    
Output:
    - Web server at configured host:port

Usage:
    python scripts/start_visualizer.py
    python scripts/start_visualizer.py --port 8000
    python scripts/start_visualizer.py --host 0.0.0.0 --port 8080
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging.logger import setup_logger, get_logger
from common.config import config

logger = get_logger("visualizer")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Start Visualizer - Web UI for Atlas exploration"
    )
    
    # All params default to config.json values
    parser.add_argument(
        "--host",
        default=config.get("visualizer.host", "localhost"),
        help=f"Host to bind (default: {config.get('visualizer.host', 'localhost')})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.get("visualizer.port", 8080),
        help=f"Port to listen (default: {config.get('visualizer.port', 8080)})"
    )
    parser.add_argument(
        "--static-dir",
        default=config.get("visualizer.static_dir"),
        help="Static files directory"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger("visualizer", console_output=True)
    
    logger.info("=" * 60)
    logger.info("PHASE 3: VISUALIZER")
    logger.info("=" * 60)
    logger.info(f"Server: http://{args.host}:{args.port}")
    logger.info("=" * 60)
    
    # Import and start
    from visualizer.server import AtlasServer
    
    server = AtlasServer(
        host=args.host,
        port=args.port,
        static_dir=args.static_dir
    )
    
    server.start()


if __name__ == "__main__":
    main()
