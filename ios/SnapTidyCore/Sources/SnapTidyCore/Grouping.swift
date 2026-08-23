import Foundation

/// 一张候选照片的元数据（分组引擎的输入抽象）。
/// 由上层（PhotosKit 层）从 PHAsset + 缩略图计算好后传入。
public struct PhotoCandidate: Sendable, Equatable {
    public let id: String            // 唯一标识（PHAsset localIdentifier）
    public let timestamp: TimeInterval // 拍摄时间（epoch seconds）
    public let dHash: UInt64         // 64-bit 感知哈希
    public let quality: Double       // 质量分 0-1（清晰度/分辨率综合，上层算好）
    public let pixelWidth: Int
    public let pixelHeight: Int

    public init(id: String, timestamp: TimeInterval, dHash: UInt64,
                quality: Double, pixelWidth: Int, pixelHeight: Int) {
        self.id = id
        self.timestamp = timestamp
        self.dHash = dHash
        self.quality = quality
        self.pixelWidth = pixelWidth
        self.pixelHeight = pixelHeight
    }
}

/// 一组相似照片 + 组内最佳。
public struct PhotoGroup: Sendable, Equatable {
    public let photos: [PhotoCandidate] // 时间升序
    public let best: PhotoCandidate     // 建议保留的一张
    public let bestIndex: Int           // best 在 photos 中的下标

    public init(photos: [PhotoCandidate], best: PhotoCandidate, bestIndex: Int) {
        self.photos = photos
        self.best = best
        self.bestIndex = bestIndex
    }
}

/// 分组引擎：把海量照片归成"相似簇"，每簇选出最佳。
///
/// 两阶段分桶（保证几万张照片在线性/近线性时间内完成）：
/// 1. **时间桶**：拍摄时间相邻间隔超过 `timeGap` 则断桶——
///    同一场景的连拍/重复天然落在同一时间窗内。
/// 2. **hash 桶**：每个时间桶内按 dHash 的前 `hashPrefixBits` 位分桶，
///    相似图必然落同一桶，桶内规模被限制在可两两比较的量级。
///
/// 桶内用并查集合并汉明距离 ≤ `maxHamming` 的相似图（传递合并，
/// 支持 A~B、B~C 但 A~C 略远的链式相似）。
public enum GroupingEngine {

    public struct Config: Sendable {
        /// 时间桶断点：相邻照片时间差超过该值则新桶（秒）
        public var timeGap: TimeInterval
        /// hash 分桶用的前缀位数（越大桶越细，相似图可能被拆开）
        public var hashPrefixBits: Int
        /// 判定"相似"的汉明距离上限
        public var maxHamming: Int

        public init(timeGap: TimeInterval = 180,
                    hashPrefixBits: Int = 14,
                    maxHamming: Int = 12) {
            self.timeGap = timeGap
            self.hashPrefixBits = hashPrefixBits
            self.maxHamming = maxHamming
        }
    }

    /// 对照片分组，返回全部组（含单张组，由上层决定哪些值得展示）。
    public static func group(_ photos: [PhotoCandidate],
                             config: Config = Config()) -> [PhotoGroup] {
        guard !photos.isEmpty else { return [] }
        let sorted = photos.sorted { $0.timestamp < $1.timestamp }

        // —— 阶段 1：时间桶 ——
        var timeBuckets: [[PhotoCandidate]] = []
        var current: [PhotoCandidate] = [sorted[0]]
        for i in 1..<sorted.count {
            if sorted[i].timestamp - sorted[i - 1].timestamp > config.timeGap {
                timeBuckets.append(current)
                current = []
            }
            current.append(sorted[i])
        }
        if !current.isEmpty { timeBuckets.append(current) }

        // —— 阶段 2：hash 桶 + 并查集合并 ——
        var groups: [PhotoGroup] = []
        for bucket in timeBuckets {
            // 按 hash 前缀分桶
            let prefixMask: UInt64 = (config.hashPrefixBits >= 64)
                ? ~0
                : ((UInt64(1) << UInt64(config.hashPrefixBits)) - 1) << UInt64(64 - config.hashPrefixBits)
            var hashBuckets: [UInt64: [Int]] = [:] // 前缀 -> 桶内下标（相对 bucket）
            for (i, p) in bucket.enumerated() {
                let prefix = p.dHash & prefixMask
                hashBuckets[prefix, default: []].append(i)
            }

            for (_ , indices) in hashBuckets {
                // 单元素桶直接成单张组（无重复照片也必须出现在输出里，由上层决定是否展示）
                if indices.count == 1 {
                    groups.append(makeGroup(indices, from: bucket))
                    continue
                }
                // 桶内两两比较 + 并查集
                var uf = UnionFind(bucket.count)
                for i in 0..<indices.count {
                    for j in (i + 1)..<indices.count {
                        let a = bucket[indices[i]], b = bucket[indices[j]]
                        if DHash.hamming(a.dHash, b.dHash) <= config.maxHamming {
                            uf.union(indices[i], indices[j])
                        }
                    }
                }
                // 收集集合
                var sets: [Int: [Int]] = [:]
                for i in indices {
                    let r = uf.find(i)
                    sets[r, default: []].append(i)
                }
                for (_, members) in sets {
                    groups.append(makeGroup(members, from: bucket))
                }
            }
        }

        // 稳定排序：按组内最早时间升序输出
        return groups.sorted {
            let t0 = $0.photos.first?.timestamp ?? 0
            let t1 = $1.photos.first?.timestamp ?? 0
            if t0 == t1 { return $0.photos.count > $1.photos.count }
            return t0 < t1
        }
    }

    private static func makeGroup(_ members: [Int], from bucket: [PhotoCandidate]) -> PhotoGroup {
        var sorted = members.map { bucket[$0] }.sorted { $0.timestamp < $1.timestamp }
        // 组内最佳：quality 最高，并列时分辨率更高者胜
        var bi = 0
        for i in 1..<sorted.count {
            let a = sorted[bi], b = sorted[i]
            let bRes = b.pixelWidth * b.pixelHeight
            let aRes = a.pixelWidth * a.pixelHeight
            if b.quality > a.quality || (b.quality == a.quality && bRes > aRes) {
                bi = i
            }
        }
        let best = sorted[bi]
        // 把 best 移到下标 0，保持组内其余按时间序
        if bi != 0 {
            let b = sorted.remove(at: bi)
            sorted.insert(b, at: 0)
            bi = 0
        }
        return PhotoGroup(photos: sorted, best: best, bestIndex: bi)
    }
}

// MARK: - 并查集（小规模，用于桶内相似合并）

private struct UnionFind {
    private var parent: [Int]

    init(_ n: Int) {
        parent = Array(0..<n)
    }

    mutating func find(_ i: Int) -> Int {
        var r = i
        while parent[r] != r { r = parent[r] }
        // 路径压缩
        var c = i
        while parent[c] != c {
            let next = parent[c]
            parent[c] = r
            c = next
        }
        return r
    }

    mutating func union(_ a: Int, _ b: Int) {
        let ra = find(a), rb = find(b)
        if ra != rb { parent[ra] = rb }
    }
}
