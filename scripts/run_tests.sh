#!/usr/bin/env bash

# 任意命令失败、使用未定义变量或管道中间失败时立即终止脚本。
set -euo pipefail

# 以脚本所在位置计算仓库根目录，使脚本不依赖调用者当前工作目录。
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 运行 CTest，并在失败时打印被测程序的完整输出。
ctest --test-dir "${repo_root}/build" --output-on-failure
