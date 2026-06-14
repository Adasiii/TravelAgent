from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from travel_agent.config import settings
from travel_agent.tools.mcp_client import BailianMCPClient


def main() -> None:
    parser = argparse.ArgumentParser(description="List tools from a Bailian MCP Streamable HTTP endpoint.")
    parser.add_argument("--endpoint", default=settings.mcp_web_search_endpoint)
    args = parser.parse_args()

    result = BailianMCPClient(args.endpoint).list_tools()
    print(result)


if __name__ == "__main__":
    main()
