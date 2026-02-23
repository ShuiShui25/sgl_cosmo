# 本地修改测试并推送到 GitHub 的推荐流程

适用仓库：`/home/geng/Codes/sgl_cosmo`

## 1. 开始前先同步状态

```bash
cd /home/geng/Codes/sgl_cosmo
git status
git branch --show-current
git pull --rebase origin main
```

说明：
- 先确认当前分支（通常是 `main`）。
- 在开始改代码前先同步远端，减少冲突。

## 2. 修改代码（只改需要改的内容）

建议：
- 优先修改代码文件，不要把临时输出、测试结果、大数据文件加入仓库。
- 保持 `.gitignore` 生效；如果新增了新的输出目录/大文件类型，先更新 `.gitignore`。

## 3. 本地测试（至少做最小验证）

根据你这次改动，选择合适的测试方式：

```bash
# 示例：语法检查（Python）
python -m py_compile path/to/file.py

# 示例：运行脚本做最小验证
bash path/to/run_script.sh
```

建议记录：
- 改了什么
- 用什么命令验证
- 验证结果是否通过

## 4. 检查改动范围（避免误提交）

```bash
git status --short
git diff
```

如果只想提交部分文件：

```bash
git add path/to/file1 path/to/file2
```

如果确认全部都应该提交：

```bash
git add -A
```

## 5. 提交前再次检查暂存区

```bash
git diff --cached --name-only
git diff --cached
```

重点确认：
- 没有把 `output/`、`data/`、`.npy/.npz/.h5` 等结果文件提交进去
- 没有把 `.DS_Store`、`._*` 等垃圾文件提交进去

## 6. 写清晰的 commit 注释（commit message）

推荐格式（简单实用版）：

```text
<类型>: <一句话说明改了什么>
```

常用类型：
- `feat`: 新功能
- `fix`: 修复问题
- `refactor`: 重构（不改功能）
- `docs`: 文档更新
- `test`: 测试相关
- `chore`: 杂项/维护

示例：

```bash
git commit -m "fix: 修正 jax01 中 w0wa 采样配置参数"
git commit -m "test: 增加 145/146 对比脚本并更新日志筛选"
git commit -m "docs: 添加本地修改测试与推送流程说明"
```

如果改动较大，建议补充正文（说明原因和验证方式）：

```bash
git commit -m "fix: 调整 DES 模拟运行脚本参数" \
           -m "原因：旧参数在部分样本上不稳定。已用短链测试验证可运行。"
```

## 7. 推送到 GitHub

```bash
git push origin main
```

首次推送新分支时：

```bash
git push -u origin <branch-name>
```

## 8. 推送后自检（建议）

```bash
git status
git log --oneline -5
```

检查点：
- 工作区是否干净（`working tree clean`）
- 最新 commit 是否已经在 `origin/main`

## 9. 常见错误避免清单（重要）

- 不要提交大数据和结果文件（尤其 `Output/`、`data/`、`*.h5`、`*.npy`、`*.npz`、`*.zip`）
- 不要在没测试的情况下直接提交重要改动
- 不要用含糊 commit 注释（如 `update`、`fix bug`）
- 推送前先确认 remote 是否是正确账号/仓库

## 10. 一套可直接复用的最小流程（速查）

```bash
cd /home/geng/Codes/sgl_cosmo
git pull --rebase origin main

# 修改代码后，运行最小测试
# python -m py_compile xxx.py

git status --short
git add <改动文件>
git diff --cached --name-only
git commit -m "fix: <本次修改说明>"
git push origin main
```

