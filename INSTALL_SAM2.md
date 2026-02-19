# SAM2 安装指南

## 问题说明

SAM2 使用 Hydra 配置系统，需要将 `sam2` 目录安装为 Python 包才能正确加载配置文件。

## 安装步骤

在项目根目录运行以下命令：

```bash
# 方法 1: 使用 pip 安装 sam2 为可编辑包
cd ~/models/iros_agent
pip install -e ./sam2

# 方法 2: 如果方法1失败，使用绝对路径
pip install -e /home/g/models/iros_agent/sam2
```

## 验证安装

```bash
python -c "import sam2; print(sam2.__file__)"
```

应该显示类似：`/home/g/models/iros_agent/sam2/__init__.py`

## 运行分割

```bash
python scripts/call_sam2_small.py --image data/R.jpg --output data/outputs/sam2_small
```

## 卸载

如果需要卸载：

```bash
pip uninstall sam2-local
```
