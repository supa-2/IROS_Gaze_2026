#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2 安装脚本
单独运行此脚本来安装 SAM2
"""

import subprocess
import sys

print("=" * 60)
print("SAM2 安装脚本")
print("=" * 60)

print("\n[*] 正在从 GitHub 安装 SAM2...")
print("    这可能需要 1-2 分钟...\n")

cmd = [sys.executable, "-m", "pip", "install",
        "git+https://github.com/facebookresearch/segment-anything-2.git"]

result = subprocess.run(cmd, text=True)

if result.returncode == 0:
    print("\n[+] SAM2 安装成功!")
    print("\n请运行以下命令验证安装:")
    print(f"  {sys.executable} -c \"import sam2; print('OK')\"")
else:
    print("\n[!] 安装失败:")
    print(result.stderr)
