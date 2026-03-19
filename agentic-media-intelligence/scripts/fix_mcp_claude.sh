#!/bin/bash
# Fix MCP tool not showing in Claude Desktop.
# Run before opening Claude if the auto-intel tool disappears after an update.

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== MCP Claude Desktop Fix ==="

# 1. Kill ghost MCP/uv processes
echo ""
echo "1. Checking for ghost processes..."
PIDS=$(pgrep -f "mcp_server.py" 2>/dev/null || true)
if [ -n "$PIDS" ]; then
  echo "   Found mcp_server processes: $PIDS"
  echo "$PIDS" | xargs kill -9 2>/dev/null || true
  echo "   Killed."
else
  echo "   No mcp_server processes found."
fi

PIDS=$(pgrep -f "uv.*mcp_server" 2>/dev/null || true)
if [ -n "$PIDS" ]; then
  echo "   Found uv+mcp processes: $PIDS"
  echo "$PIDS" | xargs kill -9 2>/dev/null || true
  echo "   Killed."
fi

# 2. Verify MCP server starts
echo ""
echo "2. Testing MCP server..."
if uv run python -c "
from app.graph.entity_workflow import build_workflow
build_workflow()
print('OK')
" 2>/dev/null; then
  echo "   MCP dependencies OK."
else
  echo "   ERROR: MCP server failed to load. Run: uv sync"
  exit 1
fi

# 3. Config check
echo ""
echo "3. Claude Desktop config should contain:"
echo "   \"auto-intel\": {"
echo "     \"command\": \"$(which uv || echo ~/.local/bin/uv)\","
echo "     \"args\": [\"--directory\", \"$PROJECT_ROOT\", \"run\", \"python\", \"src/mcp_server.py\"],"
echo "     \"env\": { \"PYTHONPATH\": \"$PROJECT_ROOT\" }"
echo "   }"
echo ""
echo "   Config path: ~/Library/Application Support/Claude/claude_desktop_config.json"
echo ""
echo "4. Next steps:"
echo "   - Quit Claude Desktop completely (Cmd+Q)"
echo "   - Wait 5 seconds"
echo "   - Reopen Claude Desktop"
echo ""
echo "Done."
