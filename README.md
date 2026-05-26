# nano-rag

本地知识库 RAG 问答系统，支持微信机器人 + Web 前端。

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 特性

- **RAG 全链路**：文档解析 → 语义分块 → 向量化入库 → 检索 → LLM 生成
- **混合检索**：BM25 稀疏检索 + 稠密向量检索，通过 RRF（Reciprocal Rank Fusion）融合排序
- **父子分块策略**：2048 字符父块保留完整上下文，256 字符子块用于精细检索，命中后自动展开为父块
- **多格式支持**：PDF（含 OCR 回退）、Markdown、TXT、代码文件（20+ 语言）
- **微信机器人**：基于 ilinkai 协议，扫码登录，长轮询消息同步，自动回复
- **Web 前端**：Markdown 渲染 + LaTeX 公式（KaTeX）+ 拖拽上传 + 知识库管理
- **零配置运行**：本地 ChromaDB 向量库，无需外部数据库
- **中文优化**：jieba 分词、中英文自动间距、中文 embedding 模型、结构化 Prompt

## 架构

```
用户（微信 / Web）
       │
       ▼
┌─────────────┐    ┌──────────────────────────────────┐    ┌──────────────┐
│   channels   │───▶│          rag/query.py            │───▶│  LLM API      │
│  wechat.py   │    │                                  │    │  生成回答      │
└─────────────┘    │  ┌─────────┐    ┌──────────┐    │    └──────────────┘
                   │  │ BM25    │    │ Dense    │    │
                   │  │ 稀疏检索 │───▶│ RRF 融合 │────┤
                   │  │ (jieba) │    │ 排序     │    │
                   │  └────┬────┘    └────┬─────┘    │
                   └───────┼──────────────┼───────────┘
                           │              │
                   ┌───────▼────┐  ┌──────▼────────┐
                   │ BM25 索引   │  │  ChromaDB     │
                   │ 关键词匹配  │  │  向量相似度    │
                   └───────▲────┘  └──────▲────────┘
                           │              │
                   ┌───────┴──────────────┴───────────┐
                   │         rag/ingest.py            │
                   │  文档解析 + 分块 + 向量化          │
                   │  (父子分块策略可选)                │
                   └──────────────────────────────────┘
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 创建配置（填入你的 LLM API Key，支持 DeepSeek / 通义千问 / OpenAI 等）
cp config.example.json config.json
# 编辑 config.json，设置 llm.active 选择厂商，填入对应的 api_key

# 3. 导入知识库
python main.py ingest knowledge/      # 导入目录
python main.py ingest 论文.pdf         # 导入单个文件

# 4. 启动服务
python main.py serve
# 打开 http://127.0.0.1:8899

# 5. 微信机器人（可选）
python main.py wechat-login   # 扫码登录
# 之后运行 python main.py serve 会自动启动微信
```

## 配置

```json
{
  "llm": {
    "active": "deepseek",          // 可选 deepseek / dashscope / openai
    "providers": {
      "deepseek": {
        "api_key": "sk-your-key",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com"
      },
      "dashscope": {
        "api_key": "sk-your-key",
        "model": "qwen-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
      },
      "openai": {
        "api_key": "sk-your-key",
        "model": "gpt-4o",
        "base_url": "https://api.openai.com/v1"
      }
    }
  },
  "rag": {
    "chunking_strategy": "parent_child",
    "parent_chunk_size": 2048,
    "child_chunk_size": 256,
    "child_chunk_overlap": 50,
    "hybrid_search": true,
    "top_k": 5,
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "persist_dir": "./chroma_db"
  },
  "wechat": {
    "enabled": true,
    "api_base": "https://ilinkai.weixin.qq.com"
  },
  "server": {
    "host": "127.0.0.1",
    "port": 8899
  }
}
```

### LLM 厂商

支持任意 OpenAI 兼容 API。在 `llm.providers` 中添加配置，将 `llm.active` 设为要使用的厂商名称即可切换。

### 分块策略

- **`flat`**（默认）：固定大小分块，简单直接
- **`parent_child`**：父子分块，检索用子块（细粒度），返回时展开为父块（完整上下文）

### 检索模式

- **`hybrid_search: false`**（默认）：仅使用稠密向量检索
- **`hybrid_search: true`**：混合检索（BM25 + 稠密向量 + RRF 融合），适合关键词精确匹配 + 语义理解场景

### 国内 HuggingFace 镜像

如果下载 embedding 模型失败，设置环境变量：

```powershell
# Windows (永久)
[Environment]::SetEnvironmentVariable('HF_ENDPOINT', 'https://hf-mirror.com', 'User')
```

```bash
# Linux / macOS
export HF_ENDPOINT=https://hf-mirror.com
```

## 命令行

```bash
python main.py ingest <path>        # 导入文件或目录
python main.py query "问题"          # 命令行问答
python main.py query "..." --retrieve-only  # 仅检索，不调 LLM
python main.py reset                # 清空知识库
python main.py serve                # 启动 Web + API + 微信
python main.py wechat-login         # 微信扫码登录
python main.py wechat-serve         # 仅启动微信机器人
```

## API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/` | GET | Web 前端 |
| `/api/health` | GET | 健康检查 |
| `/api/query` | POST | RAG 问答 `{"question": "...", "top_k": 5}` |
| `/api/kb` | GET | 知识库文件列表 |
| `/api/kb/{hash}` | DELETE | 删除文档 |
| `/api/ingest` | POST | 上传文件（multipart） |

## 扫描版 PDF OCR

PDF 无文字层时，可安装 Tesseract OCR 进行识别：

1. 安装 [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)（勾选中文语言包）
2. `pip install pytesseract pdf2image`

## 项目结构

```
nano-rag/
├── main.py              # CLI 入口 + HTTP 服务器
├── config.example.json  # 配置模板
├── requirements.txt
├── rag/
│   ├── store.py         # ChromaDB 向量库封装（单例）
│   ├── ingest.py        # 文档解析 + 分块 + 入库
│   └── query.py         # 检索 + LLM 回答
├── channels/
│   └── wechat.py        # 微信个人机器人（ilinkai 协议）
└── knowledge/           # 知识库文件目录
```

## 技术栈

| 层 | 技术 |
|---|---|
| Embedding | sentence-transformers (multilingual MiniLM) |
| 向量库 | ChromaDB (SQLite, 本地持久化) |
| 稀疏检索 | BM25 (rank-bm25) + jieba 分词 |
| 检索融合 | RRF (Reciprocal Rank Fusion) |
| LLM | OpenAI 兼容 API（DeepSeek / 通义千问 / OpenAI 等） |
| Web 框架 | aiohttp |
| 前端 | Vanilla JS + marked + KaTeX |
| PDF 解析 | pdfplumber / pypdf / Tesseract OCR |
| 微信协议 | ilinkai HTTP long-poll |
