import os
import json
from typing import List
from tooluniverse.execute_function import ToolUniverse

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

OUTPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mcp_wrappers_generated.txt"))

# Type map from JSON schema to Python types
TYPE_MAP = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "List[str]",
    "object": "dict"
}

def load_tools(file_path):
    with open(file_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f" Could not parse JSON: {file_path}")
            return []

def extract_arguments(properties):
    args = []
    for name, spec in properties.items():
        arg_type = spec.get("type", "string")
        py_type = TYPE_MAP.get(arg_type, "str")
        args.append((name, py_type))
    return args

def generate_function_code(name, args):
    if not args:
        return f"""
@mcp.tool()
def {name}() -> dict:
    return engine.run_one_function({{
        "name": "{name}",
        "arguments": {{ }}
    }})
"""
    arg_defs = ",\n    ".join([f"{k}: {v}" for k, v in args])
    arg_dict = ",\n            ".join([f'"{k}": {k}' for k, _ in args])

    return f"""
@mcp.tool()
def {name}(
    {arg_defs}
) -> dict:
    return engine.run_one_function({{
        "name": "{name}",
        "arguments": {{
            {arg_dict}
        }}
    }})
"""

def main():
    all_functions = []
    print(" Reading from:", DATA_DIR)
    print(" Files found:", os.listdir(DATA_DIR))

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(DATA_DIR, filename)
        print(f" Parsing {filename}")

        tools = load_tools(filepath)
        count = 0

        for tool in tools:
            name = tool.get("name")
            parameter = tool.get("parameter", {})
            props = parameter.get("properties") or {}
            args = extract_arguments(props)

            if name:
                all_functions.append(generate_function_code(name, args))
                count += 1

        print(f" Tools found in {filename}: {count}")

    with open(OUTPUT_FILE, "w") as f:
        f.write(
            "# Auto-generated MCP wrappers\n"
            "from fastmcp import FastMCP\n"
            "from typing import List\n"
            "from tooluniverse.execute_function import ToolUniverse\n\n"
            "mcp = FastMCP('ToolUniverse MCP', stateless_http=True)\n"
            "engine = ToolUniverse()\n"
            "engine.load_tools()\n"
        )
        f.write("\n".join(all_functions))
        f.write(
            "\n\ndef run_server():\n"
            "    mcp.run(transport='streamable-http', host='127.0.0.1', port=8000)\n\n"
            "def run_claude_desktop():\n"
            "    print(\"Starting ToolUniverse MCP server...\")\n"
            "    mcp.run(transport='stdio')\n"
        )

    print(f" Generated {len(all_functions)} MCP wrappers in {OUTPUT_FILE}")

def run_generate():
    main()

if __name__ == "__main__":
    run_generate()