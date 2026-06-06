import mlx.core as mx
import os

def load_kernel_source():
    metal_path = os.path.join(
        os.path.dirname(__file__),
        "flash_attention_hybrid.metal"
    )
    with open(metal_path, "r") as f:
        return f.read()

def metal_flash_attention_hybrid(Q, K, V):
    B, H, N, d = Q.shape
    scale = float(d ** -0.5)

    source = load_kernel_source()

    kernel = mx.fast.metal_kernel(
        name="flash_attention_hybrid_kernel",
        input_names=["Q", "K", "V", "scale"],
        output_names=["O"],
        source=source
    )

    n_query_groups = (N + 8 - 1) // 8

    outputs = kernel(
        inputs=[Q, K, V, mx.array(scale, dtype=mx.float32)],
        output_shapes=[Q.shape],
        output_dtypes=[Q.dtype],
        grid=(n_query_groups * 32, H, B),
        threadgroup=(32, 1, 1),
        template=[("N", N), ("d", d), ("H", H)]
    )

    return outputs[0]