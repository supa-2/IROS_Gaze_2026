# 快速启动指南

## 在新电脑上启动项目前的检查清单

---

### ✅ 步骤 1: 安装 Python 环境

```bash
# 检查 Python 版本 (需要 3.11+)
python --version

# 如果没有安装或版本过低，下载安装 Python 3.11+
# 官网: https://www.python.org/downloads/
```

---

### ✅ 步骤 2: 创建并激活虚拟环境

```bash
# 进入项目目录
cd IROS_AGENT

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

---

### ✅ 步骤 3: 安装依赖

```bash
# 方式 1: 使用 pip (推荐)
pip install -r requirements.txt

# 方式 2: 如果没有 requirements.txt，使用 pyproject.toml
pip install langchain langchain-openai langgraph networkx openai pandas python-dotenv matplotlib seaborn streamlit pillow torch torchvision transformers opencv-python numpy scipy openpyxl xlrd replicate
```

---

### ✅ 步骤 4: 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API 密钥
# Windows: notepad .env
# Linux: nano .env
```

**必需配置**：
```bash
# OpenAI API 或阿里云 Dashscope API
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=https://api.openai.com/v1
# 或使用阿里云
# OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 模型名称
LLM_MODEL=qwen-flash-character
```

---

### ✅ 步骤 5: 验证安装

```bash
# 测试 Python 导入
python -c "
import torch
import langchain
import networkx
print('✓ 基础依赖安装成功')
"
```

---

### ✅ 步骤 6: 运行测试

```bash
# 测试 agent
python agent.py
```

---

## 常见问题

### Q: 没有 requirements.txt？

```bash
# 手动安装核心依赖
pip install langchain>=1.2.3 langchain-openai>=1.1.7 networkx>=3.6.1 openai>=2.15.0 python-dotenv>=1.0.0
pip install pandas numpy matplotlib seaborn pillow opencv-python
```

### Q: 模块导入错误？

```bash
# 确保在项目根目录运行
cd IROS_AGENT
python agent.py
```

### Q: API 连接失败？

检查 `.env` 文件中的 API 密钥是否正确。

---

## 最小化启动

如果你想快速测试，只需：

```bash
# 1. 安装依赖
pip install langchain langchain-openai networkx openai python-dotenv

# 2. 配置 API
echo OPENAI_API_KEY=你的密钥 > .env

# 3. 运行
python agent.py
```

---

## 如果需要 SAM 2 功能

参考 `WSL_SAM2_DEPLOYMENT.md` 进行配置。
