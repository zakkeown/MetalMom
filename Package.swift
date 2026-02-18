// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MetalMom",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "MetalMomCore", targets: ["MetalMomCore"]),
        .library(name: "MetalMomBridge", type: .dynamic, targets: ["MetalMomBridge"]),
    ],
    targets: [
        .target(
            name: "MetalMomCBridge",
            dependencies: [],
            path: "Sources/MetalMomCBridge",
            publicHeadersPath: "include"
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
