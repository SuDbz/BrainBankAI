# Table of Contents
- [FastAPI](#fastapi)
  - [Key Features](#key-features)
  - [Installation](#installation)
  - [Example Usage](#example-usage)
  - [Interactive API Docs](#interactive-api-docs)
- [FastAPI-MCP](#fastapi-mcp)
  - [Key Features](#fastapi-mcp-key-features)
  - [Installation](#fastapi-mcp-installation)
  - [Example Usage](#fastapi-mcp-example-usage)
  - [Advanced Usage & Docs](#advanced-usage--docs)
  - [Requirements](#requirements)
  - [Community & Support](#community--support)
- [uv: Fast Python Package Manager](#uv-fast-python-package-manager)
  - [Highlights](#uv-highlights)
  - [Installation](#uv-installation)
  - [Example Usage](#uv-example-usage)
  - [pip vs uv Comparison](#pip-vs-uv-comparison)
- [Summary Table](#summary-table)
- [References](#references)

# FastAPI and FastAPI-MCP: Overview, Features, and Examples

## FastAPI

FastAPI is a modern, high-performance web framework for building APIs with Python, based on standard Python type hints. It is designed for speed, ease of use, and automatic generation of interactive API documentation.

### Key Features
- **High Performance**: Comparable to NodeJS and Go (built on Starlette and Pydantic)
- **Type Safety**: Uses Python type hints for validation and editor support
- **Automatic Docs**: Generates Swagger UI and ReDoc docs at `/docs` and `/redoc`
- **Async Support**: First-class support for async endpoints
- **Dependency Injection**: Powerful, easy-to-use DI system
- **Standards-Based**: Fully compatible with OpenAPI and JSON Schema

### Installation
```bash
pip install "fastapi[standard]"
pip install "uvicorn[standard]"  # For running the server
```

### Example Usage
```python
from fastapi import FastAPI
from typing import Union

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
```
Run with:
```bash
uvicorn main:app --reload
```

### Interactive API Docs
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## FastAPI-MCP

FastAPI-MCP is an extension for FastAPI that exposes your FastAPI endpoints as Model Context Protocol (MCP) tools, with built-in authentication and minimal configuration.

### Key Features
- **Authentication**: Uses FastAPI's dependency system for auth
- **Native Integration**: Not just OpenAPI-to-MCP conversion; works directly with FastAPI
- **Zero/Minimal Config**: Just point it at your FastAPI app
- **Schema Preservation**: Keeps request/response models and docs
- **Flexible Deployment**: Mount MCP server to same app or deploy separately
- **ASGI Transport**: Efficient, direct communication via ASGI

### Installation
```bash
pip install fastapi-mcp
```

### Example Usage
```python
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

app = FastAPI()

@app.get("/hello")
def hello():
    return {"message": "Hello from FastAPI-MCP!"}

mcp = FastApiMCP(app)
mcp.mount()  # Exposes endpoints at /mcp
```
Your MCP server is now available at `/mcp`.

### Advanced Usage & Docs
- See [fastapi-mcp.tadata.com](https://fastapi-mcp.tadata.com/) for full documentation
- [Examples directory](https://github.com/tadata-org/fastapi_mcp/tree/main/examples) for more code samples

### Requirements
- Python 3.10+
- FastAPI
- ASGI server (e.g., Uvicorn)

### Community & Support
- [MCParty Slack](https://join.slack.com/t/themcparty/shared_invite/zt-30yxr1zdi-2FG~XjBA0xIgYSYuKe7~Xg)
- Issues and discussions on GitHub

---

## uv: Fast Python Package Manager

[uv](https://github.com/astral-sh/uv) is an extremely fast Python package and project manager, written in Rust. It aims to replace `pip`, `pip-tools`, `pipx`, `poetry`, `pyenv`, `twine`, `virtualenv`, and more, with a single tool.

### uv Highlights
- 🚀 10-100x faster than pip
- 🗂️ Universal lockfile and workspace support
- 🐍 Installs and manages Python versions
- 🛠️ Runs and installs tools published as Python packages
- 💾 Disk-space efficient global cache
- ❇️ Inline dependency metadata for scripts
- 🖥️ Supports macOS, Linux, and Windows

### uv Installation
#### Standalone Installer (Recommended)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
#### With pip
```bash
pip install uv
```
#### With pipx
```bash
pipx install uv
```

### uv Example Usage
#### Create a new project and add dependencies
```bash
uv init example
cd example
uv add fastapi uvicorn
```
#### Run your FastAPI app
```bash
uv run uvicorn main:app --reload
```
#### Install Python versions
```bash
uv python install 3.10 3.11 3.12
uv python pin 3.12
```
#### Use pip-compatible commands
```bash
uv pip install fastapi-mcp
uv pip sync requirements.txt
```
#### Run scripts with dependencies
```bash
echo 'import requests; print(requests.get("https://astral.sh"))' > example.py
uv add --script example.py requests
uv run example.py
```

### pip vs uv Comparison
| Feature                | pip/pip-tools/pipx | uv                      |
|------------------------|--------------------|-------------------------|
| Speed                  | Standard           | 10-100x faster          |
| Python version mgmt    | pyenv/virtualenv   | Built-in                |
| Lockfile/workspace     | poetry/pip-tools   | Universal, built-in     |
| Script deps            | Manual             | Inline metadata         |
| CLI tools              | pipx               | uv tool install/run     |
| Platform support       | All                | All                     |
| Cache                  | Basic              | Global, deduplication   |

---

## Summary Table
| Feature         | FastAPI                | FastAPI-MCP                |
|-----------------|-----------------------|----------------------------|
| API Framework   | Yes                   | Extension for FastAPI      |
| Auth Support    | Yes (Depends)         | Yes (FastAPI-native)       |
| Docs Generation | Swagger/ReDoc         | Preserves FastAPI docs     |
| MCP Integration | No                    | Yes                        |
| ASGI Support    | Yes                   | Yes                        |
| Deployment      | Uvicorn, etc.         | Mount or separate          |

---

## References
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FastAPI GitHub](https://github.com/fastapi/fastapi)
- [FastAPI-MCP Documentation](https://fastapi-mcp.tadata.com/)
- [FastAPI-MCP GitHub](https://github.com/tadata-org/fastapi_mcp)
