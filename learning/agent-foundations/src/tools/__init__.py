"""4 mock tools for agent-foundations capstone."""
from .calculator import calculator_tool
from .search_mock import search_mock_tool
from .file_op import file_op_tool
from .web_mock import web_mock_tool

ALL_TOOLS = {
    "calculator": calculator_tool,
    "search_mock": search_mock_tool,
    "file_op": file_op_tool,
    "web_mock": web_mock_tool,
}

__all__ = ["calculator_tool", "search_mock_tool", "file_op_tool", "web_mock_tool", "ALL_TOOLS"]
