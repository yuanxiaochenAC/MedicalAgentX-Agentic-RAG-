# Adding New Tools to ToolUniverse
---


### Directory Structure

```
ToolUniverse/
├── README.md
├── pyproject.toml
└── src/
    └── tooluniverse/
        ├── execute_function.py
        ├── base_tool.py
        ├── mcp_server.py
        └── data/
```

---

## Adding support to new tools MCP Wrapper Generator

 After adding a new tool, you can automatically generate Python MCP wrapper functions for your tools using:

```bash
generate-mcp-tools
```
This will create a file mcp_wrappers_generated.txt with all the tool functions based on the JSON files in src/tooluniverse. You need to save it as the `mcp_wrappers_generated.py`.
