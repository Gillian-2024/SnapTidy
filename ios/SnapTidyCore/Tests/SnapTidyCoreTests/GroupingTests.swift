import Testing
import SnapTidyCore

// MARK: - 测试工具

private let baseTime: TimeInterval = 1_700_000_000

/// 翻转低 n 位 → 与原 hash 汉明距离恰为 n（前缀不变，同桶）
private func perturb(_ h: UInt64, _ n: Int) -> UInt64 {
    h ^ ((UInt64(1) << UInt64(n)) - 1)
}

private func cand(_ id: String, at t: TimeInterval, hash: UInt64,
                  quality: Double = 0.5, res: Int = 100) -> PhotoCandidate {
    PhotoCandidate(id: id, timestamp: t, dHash: hash,
                   quality: quality, pixelWidth: res, pixelHeight: res)
}

@Suite("分组引擎")
struct GroupingTests {

    /// 同一场景连拍（时间相近 + hash 相近）→ 归 1 组，best 为 quality 最高者
    @Test("相似连拍归组且 best 为质量最高")
    func similarBurstGroupsTogether() {
        let base: UInt64 = 0xDEADBEEF_01234567
        let photos = [
            cand("a", at: baseTime, hash: base, quality: 0.4),
            cand("b", at: baseTime + 1, hash: perturb(base, 3), quality: 0.8),
            cand("c", at: baseTime + 2, hash: perturb(base, 4), quality: 0.6),
            cand("d", at: baseTime + 3, hash: perturb(base, 5), quality: 0.9),
        ]
        let groups = GroupingEngine.group(photos)
        #expect(groups.count == 1)
        #expect(groups[0].photos.count == 4)
        #expect(groups[0].best.id == "d")
    }

    /// 不同场景（hash 高 14 位不同 → 不同 hash 桶）→ 各自独立
    @Test("不同场景分离")
    func differentScenesSeparate() {
        let a: UInt64 = 0x0000_0000_0000_0000
        let b: UInt64 = 0x8000_0000_0000_0000 // 最高位不同 → 不同桶
        let photos = [
            cand("a1", at: baseTime, hash: a, quality: 0.5),
            cand("a2", at: baseTime + 1, hash: perturb(a, 2), quality: 0.5),
            cand("b1", at: baseTime + 2, hash: b, quality: 0.5),
            cand("b2", at: baseTime + 3, hash: perturb(b, 2), quality: 0.5),
        ]
        let groups = GroupingEngine.group(photos)
        #expect(groups.count == 2)
        #expect(Set(groups[0].photos.map(\.id)) == ["a1", "a2"])
        #expect(Set(groups[1].photos.map(\.id)) == ["b1", "b2"])
    }

    /// hash 相同但拍摄时间相隔远超 timeGap → 切分（防止不同天拍到的同场景被误并）
    @Test("时间间隔切桶")
    func timeGapSplitsBuckets() {
        let base: UInt64 = 0x0123_4567_89AB_CDEF
        let photos = [
            cand("early", at: baseTime, hash: base, quality: 0.5),
            cand("late", at: baseTime + 600, hash: base, quality: 0.5),
        ]
        let groups = GroupingEngine.group(photos)
        #expect(groups.count == 2)
    }

    @Test("空输入返回空")
    func emptyInput() {
        #expect(GroupingEngine.group([]).isEmpty)
    }

    /// 单张照片 → 1 组 1 张
    @Test("单张自成一族")
    func singlePhoto() {
        let g = GroupingEngine.group([cand("solo", at: baseTime, hash: 42)])
        #expect(g.count == 1)
        #expect(g[0].photos.count == 1)
        #expect(g[0].best.id == "solo")
    }

    /// quality 相同时，分辨率更高者胜
    @Test("质量相同分辨率更高者胜")
    func bestTieBreakResolution() {
        let base: UInt64 = 0xABCD_1234_5678_9FED
        let photos = [
            cand("low", at: baseTime, hash: perturb(base, 1), quality: 0.5, res: 100),
            cand("high", at: baseTime + 1, hash: perturb(base, 2), quality: 0.5, res: 1000),
        ]
        let g = GroupingEngine.group(photos)
        #expect(g[0].best.id == "high")
    }

    /// 链式相似：A~B、B~C 汉明小 → 并查集传递合并为 1 组
    @Test("链式相似传递合并")
    func chainSimilarityMerges() {
        let a: UInt64 = 0x1111_1111_1111_1111
        let b = perturb(a, 3)
        let c = perturb(b, 3) // A~C 汉明 6，仍 ≤ 12
        let photos = [
            cand("a", at: baseTime, hash: a, quality: 0.3),
            cand("b", at: baseTime + 1, hash: b, quality: 0.7),
            cand("c", at: baseTime + 2, hash: c, quality: 0.9),
        ]
        let g = GroupingEngine.group(photos)
        #expect(g.count == 1)
        #expect(g[0].photos.count == 3)
        #expect(g[0].best.id == "c")
    }

    /// 同一 hash 桶内两个独立相似子簇 → 正确分为 2 组
    @Test("同桶内两个子簇正确分离")
    func twoClustersInSameHashBucket() {
        let clusterA: UInt64 = 0x0000_0000_0000_0000
        let clusterB: UInt64 = 0x0000_0000_07FF_FFFF // 高 14 位同为 0；与 A 最小汉明 = 21 > 12
        let photos = [
            cand("a1", at: baseTime, hash: clusterA, quality: 0.5),
            cand("a2", at: baseTime + 1, hash: perturb(clusterA, 2), quality: 0.5),
            cand("b1", at: baseTime + 2, hash: clusterB, quality: 0.5),
            cand("b2", at: baseTime + 3, hash: perturb(clusterB, 2), quality: 0.5),
        ]
        let g = GroupingEngine.group(photos)
        #expect(g.count == 2)
        #expect(g[0].photos.count == 2)
        #expect(g[1].photos.count == 2)
    }

    /// 组内 best 置顶，其余按时间升序
    @Test("best 置顶且组内时间有序")
    func bestIndexAndOrder() {
        let base: UInt64 = 0xF00D_1234_5678_9ABC
        let photos = [
            cand("p1", at: baseTime, hash: perturb(base, 2), quality: 0.2),
            cand("p2", at: baseTime + 1, hash: perturb(base, 1), quality: 0.9),
            cand("p3", at: baseTime + 2, hash: perturb(base, 3), quality: 0.5),
        ]
        let g = GroupingEngine.group(photos)
        #expect(g[0].bestIndex == 0)
        #expect(g[0].photos[0].id == "p2")
        #expect(g[0].photos[1].id == "p1")
        #expect(g[0].photos[2].id == "p3")
    }
}
