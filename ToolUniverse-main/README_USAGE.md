# ToolUniverse Useage

ToolUniverse now supports both Python SDK and MCP-compatible service. It supports both programmatic use and an optional HTTP-based MCP server for external integrations.

---

## Installation

### Install from source

```
python -m pip install . --no-cache-dir
```

### Install from Pypi
Pip page (https://pypi.org/project/tooluniverse)

```
pip install tooluniverse
```

### Install from GitHub

You can install the ToolUniverse SDK locally from source:
```bash
pip install git+https://github.com/mims-harvard/ToolUniverse
```

## Using the SDK (Python API)

You can import and run tools directly in your Python code using the `ToolUniverse` class.

### Example

```python
from tooluniverse.execute_function import ToolUniverse

engine = ToolUniverse()
engine.load_tools()

result = engine.run_one_function({
    "name": "FDA_get_active_ingredient_info_by_drug_name",
    "arguments": {
        "drug_name": "Panadol",
        "limit": 5,
        "skip": 0
    }
})

print(result)
```

### Get all tools

You can retrieve a list of all available tool names and their descriptions using the `refresh_tool_name_desc()` method. This is useful for discovering what tools are currently loaded and what each tool does. The method returns two lists: one containing the tool names and another containing their corresponding descriptions.

Example usage:
```
from tooluniverse import ToolUniverse

tooluni = ToolUniverse()
tooluni.load_tools()
tool_name_list, tool_desc_list = tooluni.refresh_tool_name_desc()

print("Available tools:")
for name, desc in zip(tool_name_list, tool_desc_list):
    print(f"- {name}: {desc}")
```
### Function call to a tool

```
from tooluniverse import ToolUniverse
tooluni = ToolUniverse()
tooluni.load_tools()
query = {"name": "FDA_get_indications_by_drug_name", "arguments": {"drug_name": "KISUNLA"}}
tooluni.run(query)
```

**Tool Execution Format**

Each tool accepts a dictionary with two keys:
- `name`: the name of the tool function.
- `arguments`: a dictionary of arguments required by the tool.

All tool names and argument formats are available in the `mcp_server.py` source file or dynamically listed via the MCP interface.

---

## Running the MCP Server

The SDK also supports a built-in HTTP MCP server that allows access to tools via HTTP/JSON-RPC.

### Starting the Server

After installing the package, run the following command:

```bash
tooluniverse-mcp
```

This will start the server at:

```
http://localhost:9000/mcp/
```

### Making a Request

Use any HTTP client to send a JSON-RPC request to the MCP server.

#### Example (using curl):

```bash
curl -s -X POST http://localhost:9000/mcp/ -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" -d "{\"jsonrpc\":\"2.0\",\"id\":\"1\",\"method\":\"tools/call\",\"params\":{\"name\":\"FDA_get_active_ingredient_info_by_drug_name\",\"arguments\":{\"drug_name\":\"Panadol\",\"limit\":2,\"skip\":0}}}"
```

This returns a JSON-formatted result containing the tool output.