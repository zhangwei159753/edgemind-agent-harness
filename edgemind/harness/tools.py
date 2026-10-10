import json
import uuid

from .mcp_client import McpClientManager
from .store import HarnessStore


class ToolRegistry:
    def __init__(self, store: HarnessStore, manager: McpClientManager | None = None):
        self.store = store
        self.manager = manager or McpClientManager()

    def list_enabled(self) -> list[dict]:
        return self.manager.list_tools()

    def model_tools(self) -> list[dict]:
        result = []
        for tool in self.list_enabled():
            function_name = f"{tool['server_name']}__{tool['name']}"
            result.append({"type": "function", "function": {"name": function_name,
                "description": (tool["description"] or tool["display_name"])[:1000],
                "parameters": tool["input_schema"]}})
        return result

    def propose_call(self, task_id: str, run_id: str, function_name: str, arguments_json: str) -> dict:
        if "__" not in function_name:
            raise ValueError("MCP tool name must be server__tool")
        server_name, tool_name = function_name.split("__", 1)
        available = next((item for item in self.list_enabled()
                          if item["server_name"] == server_name and item["name"] == tool_name), None)
        if available is None:
            raise ValueError("模型请求了未启用的 MCP 工具")
        try:
            if len(arguments_json or "{}") > 50000:
                raise ValueError("MCP tool arguments exceed the 50 KiB limit")
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("MCP tool arguments are not valid JSON") from exc
        if not isinstance(arguments, dict):
            raise ValueError("MCP tool arguments must be a JSON object")
        proposal = {"proposal_id": str(uuid.uuid4()), "task_id": task_id, "run_id": run_id,
                    "server_name": server_name, "tool_name": tool_name, "arguments": arguments,
                    "side_effecting": available["side_effecting"]}
        self.store.create_tool_proposal(proposal)
        return proposal

    def execute_call(self, proposal_id: str, user_confirmed: bool = False) -> dict:
        proposal = self.store.get_tool_proposal(proposal_id)
        if proposal is None or proposal["status"] != "pending":
            raise ValueError("tool proposal is missing or no longer pending")
        if proposal["side_effecting"] and not user_confirmed:
            return {"status": "awaiting_confirmation", "proposal_id": proposal_id}
        result = self.manager.call_tool(proposal["server_name"], proposal["tool_name"], proposal["arguments"])
        status = "failed" if result.get("is_error") else "executed"
        self.store.finish_tool_proposal(proposal_id, status, result)
        self.store.append_event(proposal["task_id"], "mcp_tool_result",
                                {"proposal_id": proposal_id, "server_name": proposal["server_name"],
                                 "tool_name": proposal["tool_name"], "status": status,
                                 "result": result}, proposal["run_id"])
        return {"status": status, "result": result}
