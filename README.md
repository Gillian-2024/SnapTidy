# SnapTidy

AI 辅助照片清理工具 — 扫描目录、自动去重、质量评分、视觉聚类，导出结构化审核报告。只读处理、零副作用，不触碰源文件。

## 环境要求

- Python 3.10+
- pip

## 安装依赖

```bash
pip install -r requirements.txt
```

所需包：`Pillow`, `numpy`, `torch`, `openai-clip`, `hdbscan`

## 快速开始

```bash
# 处理照片目录（默认按自然周分组）
python -m snap_tidy process /path/to/photos --output report.json

# 自定义参数
python -m snap_tidy process ./my_photos \
    --window month \
    --min-cluster-size 3 \
    --quality-threshold 70 \
    --top-k 5 \
    --device cpu \
    --output final_report.json
```

## CLI 命令参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--window` | `week` | 日期粒度：天/周/月/年 |
| `--min-cluster-size` | `5` | HDBSCAN 最小簇大小 |
| `--quality-threshold` | `60` | 质量阈值，≥此分视为保留候选 |
| `--top-k` | `5` | 每组前 k 张作为保留候选 |
| `--device` | `auto` | 计算设备：cpu / mps / cuda |
| `--batch-size` | `32` | CLIP 批处理大小 |
| `--dhash-threshold` | `5` | dHash 最大汉明距离 |
| `--simhash-threshold` | `0.95` | SimHash 最低余弦相似度 |
| `--output` | stdout | 写入 JSON 报告文件路径 |
| `--verbose` | off | 开启调试日志 |

## Pipeline 架构

七阶段预计算管道：

```
扫描(EXIF) → 加载图片 → 去重(双哈希联合) → 质量评分
    → CLIP嵌入 + HDBSCAN聚类 → 日期分组 → 分配组合键 → 构建分组 → 输出JSON
```

上传后一次性算完所有中间结果，再组装最终输出。对源文件零副作用。

### 各阶段说明

| 阶段 | 方法 | 产出 |
|------|------|------|
| 扫描 | EXIF 时间提取 + 目录遍历 | PhotoRecord 列表 |
| 去重 | dHash HD ≤ 5 ∪ SimHash cosine ≥ 0.95 | 唯一图片 + 重复关系图 |
| 质量评分 | Laplacian 清晰度 + 曝光 + 尺寸 | 每张照片 0–100 分 |
| 聚类 | CLIP ViT-B/32 嵌入 + HDBSCAN | 簇 ID + prompt 标签 |
| 日期分组 | ISO 周/天/月/年桶化 | 每张照片的日期组字符串 |
| 组合键 | 日期 × 视觉分组笛卡尔积 | 稳定分组标识 |
| 构建分组 | 按组合键收集、按质量排序 | 最终 GroupInfo 列表 |

### JSON 报告结构

```json
{
  "total_photos": 150,
  "after_dedup": 120,
  "n_groups": 8,
  "elapsed_sec": 12.4,
  "summary": {"keep": 15, "archive": 3, "pending": 102},
  "groups": [
    {
      "key": "W2026-33/A",
      "date_group": "W2026-33",
      "visual_group": "A",
      "cluster_id": 0,
      "label": "海滩日落 / W2026-33 / A",
      "photos": [
        {
          "path": "/abs/path/photo_001.jpg",
          "quality_score": 92.5,
          "action": "keep",
          "position_in_group": 1
        }
      ]
    }
  ]
}
```

**Action 取值**：`keep`（自动保留）、`archive`（低质量归档）、`pending`（待审核）

## 前端审核界面

Pipeline 输出的 JSON 可直接在 `frontend/index.html` 中打开，提供交互式审核 UI：

1. 选择报告文件
2. 滑动质量阈值、筛选视觉分组
3. 每张卡片批量操作 Keep / Archive / Pending
4. 点击 Export Selection 导出精选列表

```bash
# 跑 pipeline 出报告
python -m snap_tidy process ./photos --output report.json

# 浏览器打开审核界面
open frontend/index.html
# 或用本地服务器（file:// 协议下图片加载可能受限）
cd ~/SnapTidy && python3 -m http.server 8080
```

## 安全声明

本项目永不修改或复制源文件。所有处理均为只读，唯一输出产物是 JSON 报告。部署到生产环境时请确认文件系统访问控制符合安全策略。
