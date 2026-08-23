# ADR-0001: 手机端采用全 on-device 轻量架构

- 日期：2026-08-23
- 状态：proposed
- 决策人：Gillian + Claude

## Y-Statement

In the context of SnapTidy 从桌面 Python 原型演进为真实 iOS 相册整理 App，facing 语义去重（CLIP + HDBSCAN）在 iPhone 数万张照片上不可行的性能硬墙，且服务化推理与「照片零上传」承诺冲突，we decided 采用全 on-device 轻量管线（Swift 重写 dHash + 启发式质量分 + 两阶段分桶并查集分组），over CLIP 向量 + HDBSCAN 聚类 / 服务端推理，to achieve 数万张照片秒级完成、照片永不离开设备，accepting 相似判定从语义级降级为结构级（抓连拍/重复/相似截图，不识别「同一物品不同角度」）。

## Context

- 桌面版（Python）已跑通 CLIP + HDBSCAN 语义去重；但 HDBSCAN 需 O(n²) kNN 图，iPhone 内存与算力撑不起数万张照片
- 产品承诺「零上传」→ 服务化（照片上云推理）直接违背承诺，且引入隐私与合规负担
- Photos framework 提供系统级删除安全网：`deleteAssets` 进「最近删除」，30 天可恢复
- dHash 有公开算法出处与参考实现；启发式质量分（清晰度/分辨率）可从 Python 侧原样移植

## Decision

- 技术栈：SwiftUI + Photos framework，全部推理在设备端完成
- 管线：PHAsset 枚举 → fastFormat 缩略图 → 灰度 → dHash（9×8 相邻像素比较，64 bit）→ 时间桶 + hash 前缀桶 → 桶内汉明距离并查集并组 → best 选择（质量分最高，并列看分辨率）
- 删除走 `PHPhotoLibrary.deleteAssets` → 系统「最近删除」30 天安全网，App 不提供不可逆删除
- 不移植 CLIP/HDBSCAN；去重目标收敛为连拍/重复/相似截图级

## Consequences

- 正向：零上传隐私承诺成立；秒级性能；无服务器成本；App Store 无网络权限诉求，审核负担小
- 负向：不识别语义相似（同一主题不同构图/角度）；dHash 对旋转、大幅裁剪敏感
- 后续可能触发的 ADR：若扩展到语义去重，需引入 on-device 模型（Core ML 嵌入）或重审零上传边界

## Sources

- Krawetz, 「Kind of Like That」(2013), HackerFactor — dHash 算法定义 + 汉明阈值经验（>10 大概率不同，缩放 0-2 bit 容差）: https://www.hackerfactor.com/blog/?%2Farchives%2F529-Kind-of-Like-That.html=
- Apple Support, 「If you're missing photos or videos in the Photos app」— 最近删除保留 30 天后永久删除: https://support.apple.com/en-ph/118558
- benhoyt/dhash (PyPI) — dHash 参考实现，明确用于近重复检测: https://pypi.org/project/dhash/
