#include <metal_stdlib>
using namespace metal;

// ============================================================================
// MetalMom Parallel Reduction Compute Shaders
//
// NOTE: This file is provided for reference/documentation. Since SPM does not
// automatically compile .metal files, the runtime uses the embedded string in
// MetalShaderSource.swift (compiled via device.makeLibrary(source:options:)).
//
// Each kernel uses threadgroup shared memory for tree-based reduction.
// The first GPU pass produces one partial result per threadgroup; the Swift
// host finishes the reduction on CPU (the partial array is tiny).
// ============================================================================

// ---- reduce_sum ------------------------------------------------------------
/// Parallel sum reduction. Each threadgroup writes one partial sum.
kernel void reduce_sum(
    device const float* input [[buffer(0)]],
    device float* partials     [[buffer(1)]],
    constant uint& count       [[buffer(2)]],
    uint tid    [[thread_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint gid    [[threadgroup_position_in_grid]],
    uint tgSize [[threads_per_threadgroup]])
{
    threadgroup float shared_data[256];

    // Load -- out-of-bounds threads contribute 0 (identity for sum).
    shared_data[lid] = (tid < count) ? input[tid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction in shared memory.
    for (uint s = tgSize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_data[lid] += shared_data[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partials[gid] = shared_data[0];
    }
}

// ---- reduce_max ------------------------------------------------------------
/// Parallel max reduction. Each threadgroup writes one partial max.
kernel void reduce_max(
    device const float* input [[buffer(0)]],
    device float* partials     [[buffer(1)]],
    constant uint& count       [[buffer(2)]],
    uint tid    [[thread_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint gid    [[threadgroup_position_in_grid]],
    uint tgSize [[threads_per_threadgroup]])
{
    threadgroup float shared_data[256];

    shared_data[lid] = (tid < count) ? input[tid] : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tgSize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_data[lid] = max(shared_data[lid], shared_data[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partials[gid] = shared_data[0];
    }
}

// ---- reduce_min ------------------------------------------------------------
/// Parallel min reduction. Each threadgroup writes one partial min.
kernel void reduce_min(
    device const float* input [[buffer(0)]],
    device float* partials     [[buffer(1)]],
    constant uint& count       [[buffer(2)]],
    uint tid    [[thread_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint gid    [[threadgroup_position_in_grid]],
    uint tgSize [[threads_per_threadgroup]])
{
    threadgroup float shared_data[256];

    shared_data[lid] = (tid < count) ? input[tid] : INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tgSize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_data[lid] = min(shared_data[lid], shared_data[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partials[gid] = shared_data[0];
    }
}

// ---- reduce_argmax ---------------------------------------------------------
/// Parallel argmax reduction. Tracks value + index per element.
/// Each threadgroup writes one partial (value, index) pair.
kernel void reduce_argmax(
    device const float* input       [[buffer(0)]],
    device float* partialVals       [[buffer(1)]],
    device uint*  partialIndices    [[buffer(2)]],
    constant uint& count            [[buffer(3)]],
    uint tid    [[thread_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint gid    [[threadgroup_position_in_grid]],
    uint tgSize [[threads_per_threadgroup]])
{
    threadgroup float sharedVals[256];
    threadgroup uint  sharedIdx[256];

    sharedVals[lid] = (tid < count) ? input[tid] : -INFINITY;
    sharedIdx[lid]  = tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tgSize / 2; s > 0; s >>= 1) {
        if (lid < s && sharedVals[lid + s] > sharedVals[lid]) {
            sharedVals[lid] = sharedVals[lid + s];
            sharedIdx[lid]  = sharedIdx[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partialVals[gid]    = sharedVals[0];
        partialIndices[gid] = sharedIdx[0];
    }
}
