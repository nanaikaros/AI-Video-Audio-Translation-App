** 量化的实现
那么量化是怎么实现的呢？量化是把张量切块分组，例如每32个参数为一组。记m为这组参数中的最小值。假设我们要量化为4 bit，记d =（最大值-m）/16, 是个fp16的缩放因子。那么这组参数中的每个参数w都可以得到一个量化后的整数q, 使得w = d * q + m。这里的q是个取整后的整数。所以在量化和反量化过程中，存在取整误差导致的模型进度损失。在这个简单的例子中，32个fp16的参数需要32x2=64字节存储。量化为4bit后，32个参数需要32x4/8=16字节存储，另外还需要额外存储这组参数对应的缩放因子d和最小值m, 两个都是fp16, 共4个字节。所以量化后需要16+4=20个字节存储32个权重参数。这种w = d * q + m量化就是llama.cpp中的type-1量化。详细介绍参加https://github.com/ggerganov/llama.cpp/pull/1684。拿这里面一个量化举例：

>GGML_TYPE_Q4_K - "type-1" 4-bit quantization in super-blocks containing 8 blocks, each block > >having 32 weights. Scales and mins are quantized with 6 bits. This ends up using 4.5 bpw.

这个量化算法中，32个参数为一个block, 8个block构成一个super-block，共有32x8=256个参数。量化后每个参数量化为4bit。量化后每一组参数共用一对d和m, 都是fp16。那么一个super-block共有8x2=16个fp16。这16个fp16做成一组再量化成为6bit, 需要16x6=96bits, 加上对应这16个fp16组所需要的一对fp16的d和m共32bits。那么这16个fp16再量化后一共需要96+32 = 128bits来存储。这128bits平摊到一个super-block中256个参数，相当于每个参数需要额外的0.5bit。所以平均下来，GGML_TYPE_Q4_K量化需要4.5 bpw(bit per weight)。