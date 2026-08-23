// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SnapTidyCore",
    products: [
        .library(name: "SnapTidyCore", targets: ["SnapTidyCore"]),
    ],
    targets: [
        .target(name: "SnapTidyCore"),
        // CLT 无 XCTest/Swift Testing，用可执行 target 做断言 runner；
        // Xcode 就绪后 Tests/ 里的 @Suite 可原样跑正式测试
        .executableTarget(name: "SnapTidyCoreCheck", dependencies: ["SnapTidyCore"]),
        .testTarget(name: "SnapTidyCoreTests", dependencies: ["SnapTidyCore"]),
    ]
)
