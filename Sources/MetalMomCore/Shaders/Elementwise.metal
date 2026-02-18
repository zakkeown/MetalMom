#include <metal_stdlib>
using namespace metal;

// ============================================================================
// MetalMom Elementwise Compute Shaders
//
// NOTE: This file is provided for reference/documentation. Since SPM does not
// automatically compile .metal files, the runtime uses the embedded string in
// MetalShaderSource.swift (compiled via device.makeLibrary(source:options:)).
// ============================================================================

/// Elementwise natural log: out[i] = log(in[i])
kernel void elementwise_log(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < count) {
        output[id] = log(input[id]);
    }
}

/// Elementwise exp: out[i] = exp(in[i])
kernel void elementwise_exp(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < count) {
        output[id] = exp(input[id]);
    }
}

/// Elementwise power: out[i] = pow(in[i], exponent)
kernel void elementwise_pow(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& exponent [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < count) {
        output[id] = pow(input[id], exponent);
    }
}

/// Elementwise abs: out[i] = abs(in[i])
kernel void elementwise_abs(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < count) {
        output[id] = abs(input[id]);
    }
}

/// Amplitude to dB: out[i] = 20 * log10(max(in[i], amin))
kernel void amplitude_to_db(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& amin [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < count) {
        float val = max(input[id], amin);
        output[id] = 20.0f * log10(val);
    }
}

/// Power to dB: out[i] = 10 * log10(max(in[i], amin))
kernel void power_to_db(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& amin [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < count) {
        float val = max(input[id], amin);
        output[id] = 10.0f * log10(val);
    }
}

/// Elementwise log1p: out[i] = log(1 + in[i])
kernel void elementwise_log1p(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < count) {
        output[id] = log(1.0f + input[id]);
    }
}

/// Elementwise scale+bias: out[i] = in[i] * scale + bias
kernel void elementwise_scale_bias(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    constant float& bias [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id < count) {
        output[id] = input[id] * scale + bias;
    }
}
