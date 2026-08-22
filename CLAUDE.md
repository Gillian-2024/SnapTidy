# SnapTidy 项目宪法

## 安全铁律

- **永不触碰真实照片库**：所有测试必须使用合成图片（`tests/validation/` 下的 tiny PNG）
- 本仓库托管在用户个人 GitHub，本机为工作电脑 — 不提交任何敏感文件
- 只读处理 + JSON 报告，零副作用

## 关键路径

| 路径 | 用途 |
|------|------|
| `snap_tidy/pipeline/` | Pipeline 核心（去重/评分/聚类/分组） |
| `snap_tidy/cli.py` | CLI 入口 (`python -m snap_tidy`) |
| `snap_tidy/tests/test_pipeline.py` | 冒烟测试（5项端到端） |
| `frontend/index.html` | 审核界面（纯 HTML，双击即用） |
| `docs/phase-b-design.md` | Phase B 设计文档 |

## 开发约定

- Python 3.10+，类型标注优先（`from __future__ import annotations`）
- 不新增 pip 依赖 — Pillow/numpy/torch/openai-clip/hdbscan 已是硬依赖，新功能用这些包解决
- 每次改动后跑 `python3 -c "..."` 验证导入和冒烟测试
- 修改文件前不读全文件也没关系 — Edit 工具会精确匹配字符串

## 已知限制

- macOS quarantine 隔离属性会阻止文件访问，需要手动 `xattr -dr`
- CLIP 推理默认自动检测 device，CPU 很慢但不报错
- 前端 JSON 中的 photo path 是绝对路径，file:// 协议下可直接加载图片
