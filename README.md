# Agentic RAG 项目

## 简介

本项目是一个基于 LangGraph 实现的 **Agentic RAG** (Retrieval-Augmented Generation) 应用。它不仅能根据用户问题从本地知识库中检索相关信息，还具备自我修正的能力。如果初步检索到的文档不相关，它会尝试改写问题并进行新一轮的检索，从而提高最终生成答案的准确性和相关性。

整个项目被构建为一个可由 `langgraph dev` 启动的本地服务，方便开发和调试。

## 主要功能

- **本地文档检索**: 从本地 `datasets` 目录下的 Markdown 文件中提取知识。
- **智能决策**: Agent能够判断是直接回答问题，还是需要先进行信息检索。
- **相关性评估**: 对检索到的文档进行相关性打分，判断内容是否能回答用户的问题。
- **自我修正**: 如果检索到的文档不相关，Agent会自动改写（Re-write）用户的问题，以期获得更好的检索结果。
- **可配置模型**: 支持通过 `.env` 文件配置使用本地或任何兼容 OpenAI API 的大语言模型（LLM）。
- **服务化部署**: 可以通过 `langgraph dev` 命令快速启动一个本地 API 服务，并使用 LangGraph Studio 进行可视化调试。

## 技术栈

- **核心框架**: LangGraph, LangChain
- **模型服务**: 兼容 OpenAI API 的任意模型服务
- **环境与包管理**: `uv`
- **数据源**: 本地 Markdown 文件

## 快速开始

请遵循以下步骤来安装、配置并运行本项目。

### 1. 克隆项目

首先，将本项目克隆到您的本地机器：

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. 创建并激活虚拟环境

我们使用 `uv` 来管理虚拟环境，以确保项目依赖的隔离。

```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境 (Linux / macOS)
source .venv/bin/activate

# 激活虚拟环境 (Windows)
# .venv\Scripts\activate
```

### 3. 安装依赖

在激活虚拟环境后，使用 `uv` 来安装项目所需的所有依赖。

```bash
uv pip install -e .
```
这个命令会读取 `pyproject.toml` 文件，安装所有依赖，并将当前项目以“可编辑”模式安装。

### 4. 配置环境变量

项目需要通过环境变量来配置您的模型服务。

a. 首先，复制示例文件 `.env.example` 并重命名为 `.env`：

```bash
cp .env.example .env
```

b. 然后，编辑 `.env` 文件，填入您的本地模型服务信息：

```
# 您的本地 OpenAI 兼容 API 的基础 URL
OPENAI_API_BASE="http://localhost:8000/v1"

# 您的 API 密钥 (如果本地服务不需要，可以填写任意字符)
OPENAI_API_KEY="YOUR_API_KEY"

# 您使用的 Embedding 模型的名称
EMBEDDING_MODEL_NAME="text-embedding-ada-002"

# 您使用的 Chat 模型的名称
CHAT_MODEL_NAME="gpt-4"
```

### 5. 准备您的知识库

将您自己的 `.md` (Markdown) 文件放入项目根目录下的 `datasets` 文件夹中。您可以删除或替换掉里面自带的 `sample.md` 文件。

### 6. 启动服务

一切准备就绪！现在，使用 `langgraph` 命令行工具来启动本地开发服务。

```bash
langgraph dev
```

如果一切顺利，您会看到类似以下的输出：
```
>    Ready!
>
>    - API: http://localhost:2024/
>
>    - Docs: http://localhost:2024/docs
>
>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

## 如何使用

服务启动后，您可以通过以下方式与 Agentic RAG 应用进行交互：

1.  **LangGraph Studio (推荐)**:
    - 打开浏览器，访问 `langgraph dev` 命令输出的 `LangGraph Studio Web UI` 链接。
    - 在 Studio 界面中，您可以直观地看到图（Graph）的结构，发送请求，并实时观察每一步的输入和输出，非常适合调试。

2.  **API 请求**:
    - 您也可以使用任何 HTTP 客户端（如 `curl`, Postman, 或者 Python `requests` 库）向 `http://localhost:2024/runs/stream` 发送 POST 请求来与应用交互。
    - 请求体格式请参考 `langgraph dev` 输出的 API 文档 (`http://localhost:2024/docs`)。
