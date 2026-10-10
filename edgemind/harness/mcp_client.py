import asyncio
import json
import os
import re
from pathlib import Path


class McpClientManager:
    """Short-lived MCP stdio clients; credentials are passed from env references only."""

    def __init__(self, config_path: str | Path | None = None):
        self.config_path = Path(config_path or os.getenv("EDGEMIND_MCP_CONFIG", "mcp_servers.json"))

    def configs(self) -> dict[str, dict]:
        if not self.config_path.is_file():
            return {}
        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        servers = payload.get("mcpServers", payload.get("servers", {}))
        return {name: cfg for name, cfg in servers.items() if cfg.get("enabled", True)}

    @staticmethod
    def _sdk():
        try:
            from mcp import Client, StdioServerParameters
        except ImportError as exc:
            raise RuntimeError("MCP 插件需要可选依赖 mcp；安装项目 mcp extra 后重启。") from exc
        return Client, StdioServerParameters

    @staticmethod
    def _environment(config: dict) -> dict[str, str]:
        result = {}
        for key, value in (config.get("env") or {}).items():
            match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", str(value))
            if match and not os.getenv(match.group(1)):
                raise ValueError(f"缺少 MCP server 所需环境变量：{match.group(1)}")
            result[str(key)] = os.getenv(match.group(1), "") if match else str(value)
        return result

    async def _open(self, config: dict):
        Client, StdioServerParameters = self._sdk()
        command = str(config.get("command") or "").strip()
        if not command:
            raise ValueError("MCP server 缺少 command")
        params = StdioServerParameters(command=command, args=[str(arg) for arg in config.get("args", [])],
                                       env=self._environment(config))
        return Client(params)

    async def alist_tools(self) -> list[dict]:
        available = []
        for server_name, config in self.configs().items():
            result = await asyncio.wait_for(self._list_server_tools(config), timeout=30)
            allow = config.get("tools")
            for tool in result.tools:
                if allow is not None and tool.name not in allow:
                    continue
                available.append({"server_name": server_name, "name": tool.name,
                                  "display_name": tool.title or tool.name,
                                  "description": tool.description or "",
                                  "input_schema": tool.input_schema or {"type": "object", "properties": {}},
                                  "side_effecting": tool.name not in set(config.get("read_only_tools", []))})
        return available

    async def _list_server_tools(self, config: dict):
        client = await self._open(config)
        async with client:
            return await client.list_tools()

    async def acall_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        configs = self.configs()
        if server_name not in configs:
            raise ValueError("MCP server 未启用或未配置")
        config = configs[server_name]
        allow = config.get("tools")
        if allow is not None and tool_name not in allow:
            raise ValueError("该 MCP 工具未在允许列表中")
        result = await asyncio.wait_for(self._call_server_tool(config, tool_name, arguments), timeout=45)
        content = []
        for block in (result.content or []):
            text = getattr(block, "text", None)
            if text is not None:
                content.append(text[:10000])
            else:
                content.append({"type": getattr(block, "type", "unknown")})
        structured = result.structured_content
        if len(json.dumps(structured, ensure_ascii=False, default=str)) > 20000:
            structured = {"truncated": True, "note": "tool result exceeded 20 KiB"}
        return {"is_error": bool(result.is_error), "content": content,
                "structured_content": structured}

    async def _call_server_tool(self, config: dict, tool_name: str, arguments: dict):
        client = await self._open(config)
        async with client:
            return await client.call_tool(tool_name, arguments)

    @staticmethod
    def _run(awaitable):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        raise RuntimeError("MCP 操作需要在同步 UI worker 中运行")

    def list_tools(self) -> list[dict]:
        return self._run(self.alist_tools())

    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        return self._run(self.acall_tool(server_name, tool_name, arguments))
