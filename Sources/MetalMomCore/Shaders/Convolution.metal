#include <metal_stdlib>
using namespace metal;

// ============================================================================
// MetalMom Convolution Compute Shaders
//
// NOTE: This file is provided for reference/documentation. Since SPM does not
// automatically compile .metal files, the runtime uses the embedded string in
// MetalShaderSource.swift (compiled via device.makeLibrary(source:options:)).
//
// These implement cross-correlation (no kernel flip), matching standard
// ML/DSP convention (e.g. numpy.correlate, PyTorch conv1d).
// ============================================================================

/// 1D convolution (valid mode): output[i] = sum(input[i+k] * kernel[k])
/// Output length = inputLen - kernelSize + 1
kernel void conv1d(
    device const float* input          [[buffer(0)]],
    device const float* kernel_weights [[buffer(1)]],
    device float* output               [[buffer(2)]],
    constant uint& inputLen            [[buffer(3)]],
    constant uint& kernelSize          [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    uint outputLen = inputLen - kernelSize + 1;
    if (id >= outputLen) return;

    float sum = 0.0f;
    for (uint k = 0; k < kernelSize; k++) {
        sum += input[id + k] * kernel_weights[k];
    }
    output[id] = sum;
}

/// 2D convolution on row-major [H, W] input with [kH, kW] kernel (valid mode).
/// Output shape = [H - kH + 1, W - kW + 1]
kernel void conv2d(
    device const float* input          [[buffer(0)]],
    device const float* kernel_weights [[buffer(1)]],
    device float* output               [[buffer(2)]],
    constant uint& inputH              [[buffer(3)]],
    constant uint& inputW              [[buffer(4)]],
    constant uint& kernelH             [[buffer(5)]],
    constant uint& kernelW             [[buffer(6)]],
    uint2 pos [[thread_position_in_grid]])
{
    uint outH = inputH - kernelH + 1;
    uint outW = inputW - kernelW + 1;
    if (pos.x >= outW || pos.y >= outH) return;

    float sum = 0.0f;
    for (uint ky = 0; ky < kernelH; ky++) {
        for (uint kx = 0; kx < kernelW; kx++) {
            uint iy = pos.y + ky;
            uint ix = pos.x + kx;
            sum += input[iy * inputW + ix] * kernel_weights[ky * kernelW + kx];
        }
    }
    output[pos.y * outW + pos.x] = sum;
}

/// 1D convolution with "same" padding (output length = input length).
/// Zero-pads input implicitly.
kernel void conv1d_same(
    device const float* input          [[buffer(0)]],
    device const float* kernel_weights [[buffer(1)]],
    device float* output               [[buffer(2)]],
    constant uint& inputLen            [[buffer(3)]],
    constant uint& kernelSize          [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= inputLen) return;

    int padLeft = (int)(kernelSize / 2);
    float sum = 0.0f;
    for (uint k = 0; k < kernelSize; k++) {
        int idx = (int)id - padLeft + (int)k;
        if (idx >= 0 && idx < (int)inputLen) {
            sum += input[idx] * kernel_weights[k];
        }
    }
    output[id] = sum;
}
