import Foundation

/// Embedded Metal shader source for runtime compilation.
///
/// SPM does not automatically compile .metal files into a default library.
/// Instead, we embed the shader source as a Swift string and compile at
/// runtime via `MTLDevice.makeLibrary(source:options:)`.
///
/// The canonical .metal file lives at `Sources/MetalMomCore/Shaders/Elementwise.metal`
/// for documentation and IDE tooling, but the runtime uses this embedded string.
public enum MetalShaderSource {

    /// Elementwise compute shaders: log, exp, pow, abs, dB conversions, log1p, scale+bias.
    public static let elementwise = """
    #include <metal_stdlib>
    using namespace metal;

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
    """

    /// Parallel reduction compute shaders: sum, max, min, argmax.
    ///
    /// Each kernel uses threadgroup shared memory for tree-based reduction.
    /// The first GPU pass produces one partial result per threadgroup;
    /// the Swift host finishes the reduction on CPU (the partial array is tiny).
    public static let reduction = """
    #include <metal_stdlib>
    using namespace metal;

    // ---- reduce_sum --------------------------------------------------------
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

        // Load — out-of-bounds threads contribute 0 (identity for sum).
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

    // ---- reduce_max --------------------------------------------------------
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

    // ---- reduce_min --------------------------------------------------------
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

    // ---- reduce_argmax -----------------------------------------------------
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
    """

    /// Convolution compute shaders: 1D valid, 1D same, and 2D valid.
    ///
    /// These implement cross-correlation (no kernel flip), matching standard
    /// ML/DSP convention (e.g. numpy.correlate, PyTorch conv1d).
    public static let convolution = """
    #include <metal_stdlib>
    using namespace metal;

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
    """
}
