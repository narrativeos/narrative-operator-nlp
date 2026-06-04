# API 调用指南

> 最后更新: 2026-06-04

narrative-operator-nlp 提供三种协议接口。本文档涵盖 MCP 和 gRPC 的完整调用方式，FastAPI 见 Swagger UI (`/docs`)。

## 接口一览

| 协议 | 传输 | 端口/Socket | 适用场景 |
|------|------|------------|---------|
| **MCP** | JSON-RPC / stdio | subprocess stdin/stdout | LLM/Agent/Dify/LangGraph |
| **gRPC** | Protobuf / UDS | `/tmp/narrative-operator-nlp.sock` | 内部高性能 IPC |
| **FastAPI** | HTTP/JSON | `:8000` | 开发调试 / Studio 原型 |

---

## 1. MCP (Model Context Protocol)

### 1.1 启动 MCP Server

```bash
python adapters/mcp_server.py
```

MCP server 通过 **stdio** 通信，适合作为子进程被 LLM 框架（Claude Desktop、Dify、LangGraph）管理。

### 1.2 MCP 配置（Claude Desktop）

在 `claude_desktop_config.json` 中添加：

```json
{
  "mcpServers": {
    "narrative-operator-nlp": {
      "command": "python",
      "args": ["adapters/mcp_server.py"],
      "cwd": "/path/to/narrative-operator-nlp"
    }
  }
}
```

### 1.3 MCP 协议交互流程

```
Client                           MCP Server
  │                                  │
  │ ── initialize ────────────────► │  握手
  │ ◄── serverInfo + capabilities── │
  │                                  │
  │ ── notifications/initialized ──► │  就绪通知
  │                                  │
  │ ── tools/list ────────────────► │  查询工具
  │ ◄── [analyze_text] ──────────── │
  │                                  │
  │ ── tools/call ────────────────► │  调用分析
  │     {name:"analyze_text",       │
  │      arguments:{text:"..."}}    │
  │ ◄── NSP JSON result ─────────── │
```

### 1.4 Python 调用示例

```python
import json
import subprocess

# 启动 MCP server 作为子进程
proc = subprocess.Popen(
    ["python", "adapters/mcp_server.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
)

def call(method: str, params: dict | None = None) -> dict:
    """发送 JSON-RPC 请求并读取响应"""
    req = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params:
        req["params"] = params
    proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())

# 1. 初始化
call("initialize", {"protocolVersion": "2024-11-05",
                     "clientInfo": {"name": "my-app", "version": "1.0"}})
call("notifications/initialized")

# 2. 列出可用工具
tools = call("tools/list")
print(tools["result"]["tools"][0]["name"])  # → "analyze_text"

# 3. 分析文本
result = call("tools/call", {
    "name": "analyze_text",
    "arguments": {"text": "碳钢是钢的一种。北京立方庭位于海淀区。"}
})
nsp = json.loads(result["result"]["content"][0]["text"])
print(f"{len(nsp['content']['tokens'])} tokens, "
      f"{len(nsp['content']['entities'])} entities")

proc.terminate()
```

### 1.5 MCP Tool 定义

```json
{
  "name": "analyze_text",
  "description": "Analyze raw Chinese text using HanLP NLP engine and return NSP results",
  "inputSchema": {
    "type": "object",
    "properties": {
      "text": {
        "type": "string",
        "description": "Raw text to analyze (Chinese)."
      },
      "source": {
        "type": "string",
        "description": "NLP engine identifier.",
        "default": "hanlp_v2"
      }
    },
    "required": ["text"]
  }
}
```

---

## 2. gRPC

### 2.1 生成 Protobuf Stubs（首次）

```bash
python -m grpc_tools.protoc \
    -I schemas \
    --python_out=schemas \
    --grpc_python_out=schemas \
    schemas/narrative.proto
```

生成 `schemas/narrative_pb2.py` + `schemas/narrative_pb2_grpc.py`。

### 2.2 启动 gRPC Server

```bash
# Unix Domain Socket (默认，本机零开销)
python adapters/grpc_server.py

# 自定义 socket 路径
python adapters/grpc_server.py --socket /tmp/my-nlp.sock

# 自定义线程池大小
python adapters/grpc_server.py --workers 8
```

gRPC server 默认监听 `unix:///tmp/narrative-operator-nlp.sock`。

### 2.3 Python Client 调用

```python
from adapters.grpc_client import NlpOperatorClient

# 上下文管理器（自动关闭连接）
with NlpOperatorClient("/tmp/narrative-operator-nlp.sock") as client:
    response = client.analyze("碳钢是钢的一种。北京立方庭位于海淀区。")

    # 访问结果
    print(f"source: {response.meta.source}")
    print(f"tokens: {len(response.content.tokens)}")
    print(f"entities: {len(response.content.entities)}")
    print(f"relations: {len(response.content.relations)}")

    # 遍历实体
    for entity in response.content.entities:
        print(f"  [{entity.category}] {entity.text} "
              f"span=({entity.span.start}, {entity.span.end})")

    # 遍历关系
    for rel in response.content.relations[:5]:
        print(f"  {rel.subject} --[{rel.predicate}]--> {rel.object}")
```

### 2.4 一键调用

```python
from adapters.grpc_client import analyze_via_grpc

response = analyze_via_grpc("碳钢是钢的一种。")
print(response.meta.source)
```

### 2.5 gRPC 与 Worker Runtime 集成

按照文档 `narrative-docs/architecture/runtime/README.md` 的 Sidecar 模式：

```python
# narrative-core Worker Runtime 中
from adapters.grpc_client import NlpOperatorClient

class NlpSidecar:
    """Worker Runtime 与 NLP Operator 的 gRPC 桥接"""

    def __init__(self, socket="/tmp/narrative-operator-nlp.sock"):
        self._client = NlpOperatorClient(socket)

    def analyze(self, text: str):
        """调用 NLP 算子分析文本。自动处理连接和错误。"""
        try:
            return self._client.analyze(text)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise RuntimeError("NLP Operator unavailable. Start: python adapters/grpc_server.py")
            raise

    def close(self):
        self._client.close()

# 使用
nlp = NlpSidecar()
result = nlp.analyze("待分析的文本")
nlp.close()
```

### 2.6 Protobuf 消息结构

```
AnalyzeRequest            AnalyzeResponse
├── text: string          ├── meta: NarrativeMeta
└── source: string        │   ├── source: string
                          │   ├── version: string
                          │   ├── timestamp: string
                          │   └── text_length: int32
                          └── content: NarrativeContent
                              ├── tokens: Token[]
                              │   ├── id: int32
                              │   ├── text: string
                              │   ├── pos: string
                              │   └── span: Span {start, end}
                              ├── entities: Entity[]
                              │   ├── id: string
                              │   ├── text: string
                              │   ├── category: string
                              │   ├── span: Span
                              │   ├── normalized: string
                              │   ├── source: string
                              │   └── confidence: float
                              └── relations: Relation[]
                                  ├── id: string
                                  ├── subject: string
                                  ├── predicate: string
                                  ├── object: string
                                  ├── evidence: string
                                  ├── evidence_span: Span
                                  ├── confidence: float
                                  └── source: string
```

---

## 3. FastAPI (参考)

```bash
python adapters/fastapi_app.py
```

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | `{"status":"ok"}` |
| `/analyze` | POST | `{"text":"..."}` → NSP JSON |
| `/analyze/pretty` | POST | → HanLP `to_pretty()` 文本 |
| `/analyze/dep` | POST | → `{tokens, deps}` SVG 数据 |
| `/demo` | GET | 交互式可视化页面 |
| `/docs` | GET | Swagger UI |

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"碳钢是钢的一种。"}'
```

---

## 协议选择速查

| 场景 | 协议 | 理由 |
|------|------|------|
| LLM 调用（Claude/GPT） | MCP | 标准协议，生态兼容 |
| Agent 工作流（Dify/LangGraph） | MCP | 工具注册 + 动态发现 |
| Worker Runtime ↔ NLP Operator | gRPC + UDS | 零网络开销，类型安全 |
| 手动测试 / 调试 | FastAPI | Swagger UI，curl 友好 |
| Studio 前端原型 | FastAPI | REST 简单，JSON 直接 |
| 跨语言（Rust↔Python） | gRPC | Protobuf 类型安全 |
| 离线 / 内网部署 | gRPC + UDS | 无网络依赖 |

## 相关文档

- [项目实现总结](IMPLEMENTATION.md)
- [NSP 协议标准](PROTOCOL.md)
- [HanLP 官方 Demo 差异报告](GAP-ANALYSIS.md)
