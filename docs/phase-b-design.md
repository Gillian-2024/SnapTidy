# Phase B 前端审核界面 — 设计

## 目标

消费后端产出的 JSON 报告，提供可交互的审核 UI，让用户批量决定 `keep/archive`，导出最终精选列表。

**约束**：只生成精选报告，不碰原文件。

---

## 界面骨架（三段式）

```
┌──────────────────────────────────────────────────────┐
│  Filter bar: [window ▼] [quality ▰] [cluster ▼]    │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────┐  ┌──────┐  ┌──────┐   ┌──────┐  ┌──────┐ │
│  │Best  │  │Best  │  │Best  │   │Best  │  │Best  │ │
│  │photo │  │photo │  │photo │   │photo │  │photo │ │
│  │previe │  │previe │  │previe│   │previe │  │previe│ │
│  │w + 📋│  │w + 📋│  │w + 📋│   │w + 📋│  │w + 📋│ │
│  └──────┘  └──────┘  └──────┘   └──────┘  └──────┘ │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 1. Filter bar

| 控件 | 数据来源 | 默认值 |
|------|----------|--------|
| **窗口粒度** | 用户运行时指定，不可改 | day/week/month/year |
| **质量阈值滑块** | `quality_threshold` | 60 |
| **视觉分组筛选** | HDBSCAN cluster 标签 | 全部显示 |
| **去重折叠** | `dedup_removed` | 自动展开 |

### 2. Group card — 每组一张卡片

核心元素：
- **最佳照片缩略图**（quality_score 最高者，低饱和底色指示 quality 区间）
- **组名** = `label`（如 "beach scene · W2026-33 · A"）
- **其余照片小网格**（按 score 降序排列）
- **Action bar**: 每张照片有 [✓ Keep] [🗑 Archive] [○ Pending]
- **组级快捷操作**: "全部 Keep", "全部 Archive"

### 3. Summary panel

| 指标 | 计算方式 |
|------|----------|
| 保留数 | `summary.keep` |
| 归档数 | `summary.archive` |
| 待审数 | `summary.pending` |
| 总节省空间 | `dedup_removed` 中 file_size 总和 |

---

## Action 流转逻辑

```
pending → keep | archive → locked
```

- 用户点击后状态锁定，不可回退（除非刷新页面重新加载报告）
- 低质量照片 (< threshold) 初始为 `archive`
- 高质量且 top-k 内初始为 `keep`
- 中间地带为 `pending`

---

## 导出格式

用户点"导出精选报告"时生成：

```json
{
  "exported_at": "2026-08-23T10:00:00Z",
  "source_dir": "/path/to/photos",
  "actions": {
    "keep": ["/abs/path/photo_001.jpg", ...],
    "archive": ["/abs/path/photo_050.jpg", ...],
    "pending": ["/abs/path/photo_075.jpg", ...]
  },
  "dedup_summary": {
    "kept": ["/abs/path/photo_001.jpg"],
    "removed": ["/abs/path/photo_002.jpg", ...]
  }
}
```

---

## 技术选型建议

| 方案 | 优点 | 缺点 |
|------|------|------|
| **纯 HTML + Vanilla JS** | 零依赖，双击打开即用 | 大项目代码量多 |
| **HTMX + server template** | 需要后端配合 | MVP 只需 JSON |
| **React/Vue SPA** | 生态丰富 | 需要构建步骤 |

**推荐**：纯 HTML + Vanilla JS。JSON 已结构化，不需要服务端渲染。
