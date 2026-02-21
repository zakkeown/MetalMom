// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MetalMom",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "MetalMomCore", targets: ["MetalMomCore"]),
        .library(name: "MetalMomBridge", type: .dynamic, targets: ["MetalMomBridge"]),
        .library(name: "MetalMomBridgeStatic", type: .static, targets: ["MetalMomBridge"]),
    ],
    targets: [
        .systemLibrary(
            name: "MetalMomCBridge",
            path: "Sources/MetalMomCBridge"
        ),
        .target(
            name: "MetalMomCore",
            dependencies: ["MetalMomCBridge"],
            path: "Sources/MetalMomCore",
            exclude: ["Shaders"]
        ),
        .target(
            name: "MetalMomBridge",
            dependencies: ["MetalMomCore"],
            path: "Sources/MetalMomBridge"
        ),
        .executableTarget(
            name: "ProfilingRunner",
            dependencies: ["MetalMomCore"],
            path: "Sources/ProfilingRunner"
        ),
        .testTarget(
            name: "MetalMomTests",
            dependencies: ["MetalMomCore", "MetalMomBridge"],
            path: "Tests/MetalMomTests"
        ),
    ]
)
