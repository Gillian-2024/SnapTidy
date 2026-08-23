# SnapTidy iOS 设计文档

> 对应 ADR-0001（on-device 轻量架构）。本文描述 iOS App 的模块划分、数据流与关键实现。
> 桌面版原型见仓库根 `docs/phase-b-design.md` 与 `frontend/`。

## 1. 目标与边界

**目标**：真实可用的 iOS 相册整理 App——扫描全库照片，自动归组相似连拍/重复，逐组抽卡式审阅，一键保留最佳、其余进「最近删除」。

**边界**（ADR-0001 约束）：
- 全部推理在设备端，零上传
- 相似判定为结构级（重复/连拍/相似截图），不做语义级
- 删除走系统安全网，不提供不可逆删除

## 2. 分层架构

```
┌─────────────────────────────────────────┐
│  UI 层（SwiftUI）                        │  抽卡闭环、组审阅、进度
├─────────────────────────────────────────┤
│  PhotosKit 层（PhotoLibraryService）     │  权限、枚举、缩略图、删除
├─────────────────────────────────────────┤
│  Core 层（SnapTidyCore 包）              │  纯逻辑，无系统依赖，可单元测试
│    DHash.swift / Grouping.swift / Quality │
└─────────────────────────────────────────┘
```

- **Core 层**：已作为 SwiftPM 包 `SnapTidyCore` 落地，`swift run SnapTidyCoreCheck` 全绿（25 断言）。不依赖 UIKit/Photos，可在无 Xcode 的 CLT 环境编译验证。
- **PhotosKit 层**：`PhotoLibraryService.swift`，桥接 PHAsset ↔ Core 的 `PhotoCandidate`。
- **UI 层**：SwiftUI 视图，翻译桌面原型 `frontend/index.html` 的抽卡闭环。

## 3. 数据流

```
请求授权(PHPhotoLibrary.requestAuthorization)
  → 枚举 PHAsset（fetchAssets, 排除视频/隐藏）
  → 每张取 fastFormat 缩略图 + 元数据(拍摄时间/分辨率)
  → 灰度 → DHash.compute → PhotoCandidate
  → GroupingEngine.group（时间桶 → hash 前缀桶 → 汉明并查集）
  → [PhotoGroup] 按组呈现
  → 用户逐组选 best → PHPhotoLibrary.deleteAssets(其余) → 「最近删除」安全网
```

**质量分**（Core 层 `quality`，由 PhotosKit 层计算传入）：
- 清晰度：缩略图局部梯度方差（越清晰分越高，模糊/抖动照片分低）
- 分辨率：像素数归一化
- 合成：`quality = w1·清晰度 + w2·分辨率归一化`，best 取组内最高

## 4. 关键模块 API（草案）

```swift
// PhotosKit 层
protocol PhotoLibraryServicing {
    func requestAuthorization() async -> PHAuthorizationStatus
    func loadCandidates(progress: @escaping (Int, Int) -> Void) async throws -> [PhotoCandidate]
    func thumbnail(for id: String, targetSize: CGSize) async -> UIImage?
    func delete(ids: [String]) async throws   // → 「最近删除」
}

// Core 层（已实现）
enum DHash {
    static func compute(fromGrayPixels: [UInt8], width: Int, height: Int) -> UInt64
    static func hamming(_ a: UInt64, _ b: UInt64) -> Int
}
enum GroupingEngine {
    static func group(_ photos: [PhotoCandidate], config: Config) -> [PhotoGroup]
}
```

## 5. 抽卡闭环（UI 翻译桌面原型）

每组一卡：
- 卡片显示组内全部缩略图缩略网格 + best 高亮
- 操作：保留 best / 全删 / 手动选一张保留
- 顶部进度：剩余组数、可回收张数、可回收容量（MB）
- 组内展开可对比原图

桌面原型 `frontend/index.html` 的视觉语言（深色卡片 + 高对比数字）原样搬入。

## 6. 安全铁律（仓库级，测试时强制）

- **永不触碰真实照片库**：所有测试用合成图片（`tests/validation/` tiny PNG）
- 开发期模拟器用假数据源（MockPhotoLibrary），不接真实 PHAsset
- 只读处理 + 明确确认后才删除；删除永远进「最近删除」
- 本仓库托管在个人 GitHub，不提交任何敏感文件

## 7. 里程碑

| # | 内容 | 状态 |
|---|---|---|
| 1 | SwiftPM 包 + 分组引擎（dHash/时间桶/hash桶/并查集） | ✅ `swift run` 25/25 |
| 2 | ADR-0001 + 本设计文档 | ✅ 落盘 |
| 3 | PhotosKit 层（权限/枚举/缩略图/删除） | ⏳ 待 Xcode 就绪 |
| 4 | SwiftUI UI（抽卡闭环） | ⏳ |
| 5 | Xcode 工程 + 真机验证 | ⏳ |

> 环境注记：本机 CLT 无 XCTest/Swift Testing，Core 层用可执行断言 runner（`SnapTidyCoreCheck`）验证；Xcode 就绪后 `Tests/` 里同语义的 `@Suite` 原样跑正式测试。
