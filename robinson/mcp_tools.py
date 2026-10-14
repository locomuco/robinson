"""Robinson's tools as a stdio MCP server, for online backends (Claude Code).

Same functions and the same memory.json as the local backend, so both
models share one memory. Run as `python -m robinson.mcp_tools`.
"""

import logging
import os

from mcp.server.mcpserver import MCPServer

from .config import Config
from .eyecal import EyeCalibration
from .memory import Memory
from .tools import make_tools


def main() -> None:
    logging.basicConfig(level=logging.WARNING)   # stdout belongs to the protocol
    cfg = Config.from_env()
    memory = Memory(os.environ.get("ROBINSON_MEMORY_FILE", cfg.memory_file))
    server = MCPServer("robinson", instructions="Werkzeuge des Roboters Robinson.")
    eyes = EyeCalibration(os.environ.get("ROBINSON_EYES_FILE", cfg.eyes_file))
    for fn in make_tools(memory, eyes):
        server.add_tool(fn)
    server.run("stdio")


if __name__ == "__main__":
    main()
