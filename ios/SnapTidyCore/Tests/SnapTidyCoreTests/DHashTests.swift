import Testing
import SnapTidyCore

@Suite("DHash")
struct DHashTests {

    /// 全等灰度 → 无相邻差异 → hash 为 0
    @Test("全等灰度 hash 为 0")
    func pureColorZeroHash() {
        let pixels = [UInt8](repeating: 100, count: 9 * 8)
        #expect(DHash.compute(fromGrayPixels: pixels, width: 9, height: 8) == 0)
    }

    /// 构造 9×8 已知图案，验证精确 bit。
    /// 第 0 行递增（相邻左<右 → bit=0），第 1 行递减（左>右 → bit=1 → bits 8..15 置位）。
    @Test("已知梯度图案 hash 精确匹配")
    func knownGradientHash() {
        var pixels = [UInt8](repeating: 0, count: 9 * 8)
        for x in 0..<9 { pixels[x] = UInt8(x * 10) }          // 行 0 递增
        for x in 0..<9 { pixels[9 + x] = UInt8(80 - x * 10) } // 行 1 递减

        let hash = DHash.compute(fromGrayPixels: pixels, width: 9, height: 8)
        #expect(hash == 0xFF << 8)
    }

    /// 相同内容不同分辨率 → 最近邻缩放下 hash 一致
    @Test("缩放不变性")
    func scaleInvariance() {
        var small = [UInt8](repeating: 0, count: 9 * 8)
        for y in 0..<8 {
            for x in 0..<9 { small[y * 9 + x] = UInt8((x + y) * 8) }
        }
        var large = [UInt8](repeating: 0, count: 90 * 80)
        for y in 0..<80 {
            for x in 0..<90 { large[y * 90 + x] = small[(y / 10) * 9 + (x / 10)] }
        }
        let hSmall = DHash.compute(fromGrayPixels: small, width: 9, height: 8)
        let hLarge = DHash.compute(fromGrayPixels: large, width: 90, height: 80)
        #expect(hSmall == hLarge)
    }

    /// 几乎相同的图案（单个像素扰动）→ 汉明距离很小
    @Test("轻微扰动汉明距离小")
    func similarImagesSmallHamming() {
        var a = [UInt8](repeating: 50, count: 9 * 8)
        var b = a
        b[20] = 51
        let ha = DHash.compute(fromGrayPixels: a, width: 9, height: 8)
        let hb = DHash.compute(fromGrayPixels: b, width: 9, height: 8)
        #expect(DHash.hamming(ha, hb) <= 2)
    }

    /// 相反梯度（全 0 vs 全 1 的 64 位模式）→ 汉明距离大
    @Test("相反梯度汉明距离大")
    func differentImagesLargeHamming() {
        var up = [UInt8](repeating: 0, count: 9 * 8)
        for y in 0..<8 {
            for x in 0..<9 { up[y * 9 + x] = UInt8(x * 30) }
        }
        var down = [UInt8](repeating: 0, count: 9 * 8)
        for y in 0..<8 {
            for x in 0..<9 { down[y * 9 + x] = UInt8((8 - x) * 30) }
        }
        let hUp = DHash.compute(fromGrayPixels: up, width: 9, height: 8)
        let hDown = DHash.compute(fromGrayPixels: down, width: 9, height: 8)
        #expect(DHash.hamming(hUp, hDown) >= 60)
    }

    /// 汉明距离是交换对称的
    @Test("汉明距离对称")
    func hammingSymmetric() {
        let a: UInt64 = 0b1010_1010_1010_1010
        let b: UInt64 = 0b1100_1100_1100_1100
        #expect(DHash.hamming(a, b) == DHash.hamming(b, a))
    }
}
