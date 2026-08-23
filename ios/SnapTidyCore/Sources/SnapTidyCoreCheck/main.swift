import Foundation
import SnapTidyCore
import Darwin

// 断言 runner：不依赖 XCTest/Swift Testing（CLT 环境无 Apple 测试框架）。
// 覆盖 DHash + GroupingEngine 全部语义。

var pass = 0
var fail = 0

@MainActor
func check(_ name: String, _ cond: @autoclosure () -> Bool) {
    if cond() { pass += 1; print("PASS  \(name)") }
    else { fail += 1; print("FAIL  \(name)") }
}

@MainActor
func checkEqual<T: Equatable>(_ name: String, _ got: T, _ want: T) {
    if got == want { pass += 1; print("PASS  \(name)") }
    else { fail += 1; print("FAIL  \(name) — got \(got), want \(want)") }
}

// MARK: - 测试工具

let baseTime: TimeInterval = 1_700_000_000

/// 翻转低 n 位 → 与原 hash 汉明距离恰为 n（高 14 位前缀不变，同桶）
func perturb(_ h: UInt64, _ n: Int) -> UInt64 {
    h ^ ((UInt64(1) << UInt64(n)) - 1)
}

func cand(_ id: String, at t: TimeInterval, hash: UInt64,
          quality: Double = 0.5, res: Int = 100) -> PhotoCandidate {
    PhotoCandidate(id: id, timestamp: t, dHash: hash,
                   quality: quality, pixelWidth: res, pixelHeight: res)
}

do {
    // 全等灰度 → hash 0
    let flat = [UInt8](repeating: 100, count: 9 * 8)
    checkEqual("DHash: 全等灰度 hash 为 0",
               DHash.compute(fromGrayPixels: flat, width: 9, height: 8), 0)

    // 已知梯度：行 0 递增(bit=0)，行 1 递减(bit=1 → bits 8..15 置位)
    var grad = [UInt8](repeating: 0, count: 9 * 8)
    for x in 0..<9 { grad[x] = UInt8(x * 10) }
    for x in 0..<9 { grad[9 + x] = UInt8(80 - x * 10) }
    checkEqual("DHash: 已知梯度精确匹配",
               DHash.compute(fromGrayPixels: grad, width: 9, height: 8), 0xFF << 8)

    // 缩放不变性：9×8 内容放大到 90×80，最近邻采样 hash 一致
    var small = [UInt8](repeating: 0, count: 9 * 8)
    for y in 0..<8 { for x in 0..<9 { small[y * 9 + x] = UInt8((x + y) * 8) } }
    var large = [UInt8](repeating: 0, count: 90 * 80)
    for y in 0..<80 { for x in 0..<90 { large[y * 90 + x] = small[(y / 10) * 9 + (x / 10)] } }
    let hs = DHash.compute(fromGrayPixels: small, width: 9, height: 8)
    let hl = DHash.compute(fromGrayPixels: large, width: 90, height: 80)
    check("DHash: 缩放不变性", hs == hl)

    // 轻微扰动 → 汉明距离小
    let a = [UInt8](repeating: 50, count: 9 * 8)
    var b = a; b[20] = 51
    let ha = DHash.compute(fromGrayPixels: a, width: 9, height: 8)
    let hb = DHash.compute(fromGrayPixels: b, width: 9, height: 8)
    check("DHash: 轻微扰动汉明 ≤ 2", DHash.hamming(ha, hb) <= 2)

    // 相反梯度 → 汉明距离大
    var up = [UInt8](repeating: 0, count: 9 * 8)
    var down = [UInt8](repeating: 0, count: 9 * 8)
    for y in 0..<8 {
        for x in 0..<9 { up[y * 9 + x] = UInt8(x * 30); down[y * 9 + x] = UInt8((8 - x) * 30) }
    }
    let hUp = DHash.compute(fromGrayPixels: up, width: 9, height: 8)
    let hDown = DHash.compute(fromGrayPixels: down, width: 9, height: 8)
    check("DHash: 相反梯度汉明 ≥ 60", DHash.hamming(hUp, hDown) >= 60)

    // 汉明对称
    let x1: UInt64 = 0b1010_1010_1010_1010
    let x2: UInt64 = 0b1100_1100_1100_1100
    check("DHash: 汉明对称", DHash.hamming(x1, x2) == DHash.hamming(x2, x1))
}

do {
    // 相似连拍归组 + best
    let base: UInt64 = 0xDEADBEEF_01234567
    let burst = [
        cand("a", at: baseTime, hash: base, quality: 0.4),
        cand("b", at: baseTime + 1, hash: perturb(base, 3), quality: 0.8),
        cand("c", at: baseTime + 2, hash: perturb(base, 4), quality: 0.6),
        cand("d", at: baseTime + 3, hash: perturb(base, 5), quality: 0.9),
    ]
    let g1 = GroupingEngine.group(burst)
    checkEqual("Group: 连拍归 1 组", g1.count, 1)
    checkEqual("Group: 组内 4 张", g1[0].photos.count, 4)
    checkEqual("Group: best 为 quality 最高", g1[0].best.id, "d")

    // 不同场景分离
    let sA: UInt64 = 0x0000_0000_0000_0000
    let sB: UInt64 = 0x8000_0000_0000_0000
    let scenes = [
        cand("a1", at: baseTime, hash: sA, quality: 0.5),
        cand("a2", at: baseTime + 1, hash: perturb(sA, 2), quality: 0.5),
        cand("b1", at: baseTime + 2, hash: sB, quality: 0.5),
        cand("b2", at: baseTime + 3, hash: perturb(sB, 2), quality: 0.5),
    ]
    let g2 = GroupingEngine.group(scenes)
    checkEqual("Group: 不同场景分 2 组", g2.count, 2)
    check("Group: 组 0 = a1/a2", Set(g2[0].photos.map { $0.id }) == Set(["a1", "a2"]))
    check("Group: 组 1 = b1/b2", Set(g2[1].photos.map { $0.id }) == Set(["b1", "b2"]))

    // 时间间隔切桶
    let timeBase: UInt64 = 0x0123_4567_89AB_CDEF
    let far = [
        cand("early", at: baseTime, hash: timeBase, quality: 0.5),
        cand("late", at: baseTime + 600, hash: timeBase, quality: 0.5),
    ]
    checkEqual("Group: 时间隔 600s 切 2 组", GroupingEngine.group(far).count, 2)

    // 空输入
    check("Group: 空输入空结果", GroupingEngine.group([]).isEmpty)

    // 单张
    let solo = GroupingEngine.group([cand("solo", at: baseTime, hash: 42)])
    checkEqual("Group: 单张 1 组", solo.count, 1)
    checkEqual("Group: 单张组成员 1", solo[0].photos.count, 1)
    checkEqual("Group: 单张 best", solo[0].best.id, "solo")

    // quality 相同 → 分辨率更高者胜
    let tieBase: UInt64 = 0xABCD_1234_5678_9FED
    let tie = [
        cand("low", at: baseTime, hash: perturb(tieBase, 1), quality: 0.5, res: 100),
        cand("high", at: baseTime + 1, hash: perturb(tieBase, 2), quality: 0.5, res: 1000),
    ]
    checkEqual("Group: 分辨率 tie-break", GroupingEngine.group(tie)[0].best.id, "high")

    // 链式相似传递合并
    let cA: UInt64 = 0x1111_1111_1111_1111
    let cB = perturb(cA, 3)
    let cC = perturb(cB, 3)
    let chain = [
        cand("a", at: baseTime, hash: cA, quality: 0.3),
        cand("b", at: baseTime + 1, hash: cB, quality: 0.7),
        cand("c", at: baseTime + 2, hash: cC, quality: 0.9),
    ]
    let g3 = GroupingEngine.group(chain)
    checkEqual("Group: 链式合并 1 组", g3.count, 1)
    checkEqual("Group: 链式 3 张", g3[0].photos.count, 3)
    checkEqual("Group: 链式 best", g3[0].best.id, "c")

    // 同桶内两个子簇分离（簇间最小汉明 21 > maxHamming 12，确保 perturb 不桥接合并）
    let clA: UInt64 = 0x0000_0000_0000_0000
    let clB: UInt64 = 0x0000_0000_07FF_FFFF
    let twoCluster = [
        cand("a1", at: baseTime, hash: clA, quality: 0.5),
        cand("a2", at: baseTime + 1, hash: perturb(clA, 2), quality: 0.5),
        cand("b1", at: baseTime + 2, hash: clB, quality: 0.5),
        cand("b2", at: baseTime + 3, hash: perturb(clB, 2), quality: 0.5),
    ]
    let g4 = GroupingEngine.group(twoCluster)
    checkEqual("Group: 同桶双子簇分 2 组", g4.count, 2)
    check("Group: 每簇 2 张", g4.allSatisfy { $0.photos.count == 2 })

    // best 置顶 + 组内时间有序
    let obBase: UInt64 = 0xF00D_1234_5678_9ABC
    let ordered = [
        cand("p1", at: baseTime, hash: perturb(obBase, 2), quality: 0.2),
        cand("p2", at: baseTime + 1, hash: perturb(obBase, 1), quality: 0.9),
        cand("p3", at: baseTime + 2, hash: perturb(obBase, 3), quality: 0.5),
    ]
    let g5 = GroupingEngine.group(ordered)
    checkEqual("Group: bestIndex 置顶", g5[0].bestIndex, 0)
    checkEqual("Group: 组内顺序 p2,p1,p3",
               g5[0].photos.map { $0.id }, ["p2", "p1", "p3"])
}

// MARK: - 汇总

print("---")
print("\(pass) PASS / \(fail) FAIL")
exit(fail == 0 ? 0 : 1)
