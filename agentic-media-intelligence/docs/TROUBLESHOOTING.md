# Troubleshooting

## MCP Tool Stops Working in Claude Desktop

If the MCP tool was working yesterday and is gone today, it is often a config path or process issue triggered by a Claude Desktop update. Claude Desktop has been pushing "MSIX virtualized" updates on Windows and "Sandbox" updates on macOS that can break the link to `claude_desktop_config.json`.

### Quick Fix: Run the Script

```bash
./scripts/fix_mcp_claude.sh
```

This kills ghost processes, verifies the MCP server loads, and prints the expected config. Then quit Claude (Cmd+Q), wait 5 seconds, and reopen.

### Manual: Kill Ghost Processes

The MCP server process (Python/uv) can hang in the background after closing Claude. On restart, Claude fails to bind to the same port or stdio pipe.

**macOS / Linux**

```bash
ps aux | grep mcp_server
ps aux | grep uv
```

Then `kill -9 <PID>` any lingering processes associated with this project.

**Windows**

Open Task Manager, find any stray `python.exe` or `uv.exe` associated with the project folder, and End Task.

### Verify Config Path

**macOS config path:** `~/Library/Application Support/Claude/claude_desktop_config.json`

After a Claude Desktop update, confirm the config is still in this location and that your MCP server entry points to the correct project path.
