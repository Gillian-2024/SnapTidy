import Foundation

/// 感知哈希（dHash）：把图像内容折叠成 64-bit 指纹，
/// 内容越相似，两个指纹的汉明距离越小。
///
/// 算法：灰度图缩放至 9×8，逐行比较相邻像素亮度，
/// 左亮于右则该位为 1，共 8 行 × 8 位 = 64 bit。
public enum DHash {

    /// 从灰度像素数组计算 64-bit dHash。
    /// - Parameters:
    ///   - pixels: `width * height` 个 0-255 灰度值（逐行存储）
    ///   - width: 图像宽度
    ///   - height: 图像高度
    /// - Returns: 64-bit 感知哈希
    public static func compute(fromGrayPixels pixels: [UInt8], width: Int, height: Int) -> UInt64 {
        precondition(pixels.count >= width * height, "pixels 数组长度不足 width*height")
        precondition(width > 0 && height > 0, "图像尺寸必须为正")

        // 缩放至 9×8（最近邻采样）
        let sw = 9, sh = 8
        var resized = [UInt8](repeating: 0, count: sw * sh)
        for y in 0..<sh {
            let sy = min(y * height / sh, height - 1)
            for x in 0..<sw {
                let sx = min(x * width / sw, width - 1)
                resized[y * sw + x] = pixels[sy * width + sx]
            }
        }

        // 每行 8 个相邻比较，行内 8 bit，共 64 bit
        var hash: UInt64 = 0
        for y in 0..<sh {
            for x in 0..<(sw - 1) {
                if resized[y * sw + x] > resized[y * sw + x + 1] {
                    hash |= 1 << UInt64(y * (sw - 1) + x)
                }
            }
        }
        return hash
    }

    /// 两个 dHash 的汉明距离（不同位的数量）。
    /// 值越小表示内容越相似；经验阈值：≤ 10 高度相似，≤ 14 可能相似。
    public static func hamming(_ a: UInt64, _ b: UInt64) -> Int {
        var v = a ^ b
        var d = 0
        while v != 0 {
            v &= v - 1
            d += 1
        }
        return d
    }
}
