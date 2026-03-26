1. LayerNorm和RMSNorm的区别是什么：
LayerNorm 会先“去均值再归一”（中心化 + 标准化）；RMSNorm 只按信号能量缩放（不去均值）。

LayerNorm：把每条样本的特征分布拉到“均值 0、方差 1”，再用可学的 γ、β 恢复尺度与偏移。
RMSNorm：把整向量按其 RMS（均方根）缩放，使尺度一致，但保留原始向量的偏移（均值）。

2. lora技术
LoRA(Low-Rank Adaptation of LLMs)，即LLMs的低秩适应，是参数高效微调最常用的方法。

秩: 矩阵的秩是矩阵中行向量或列向量的极大线性无关组的个数，反映矩阵所能表示的空间维度。


3. rope技术
在处理长文本时，AI模型常受限于训练时设定的上下文长度，导致无法完整理解书籍、代码库等长文档。llama.cpp通过RoPE（Rotary Position Embedding，旋转位置编码）缩放技术，让模型在不重新训练的情况下支持更长文本处理。

4. KV cache
在随后的迭代中，只计算最新 token 的 key 向量即可，其余的从缓存中提取，共同组成 K 矩阵，此外，新计算的 key 向量也会保存到缓存中，同样的过程也适用于 value 向量

我们之所以能够利用缓存来处理 key 和 value 向量，主要是因为这些向量在两次迭代之间保持相同。例如，如果我们先处理前四个 token，然后处理第五个 token，最初的四个 token 保持不变，那么前四个 token 的 key 和 value 向量在第一次迭代和第二次迭代之间将保持相同，因此，我们其实根本不需要在第二次迭代中重新计算前四个 token 的 key 和 value 向量

这一原则适用于 Transformer 中的所有层，而不仅仅是第一层，在所有层中，每个 token 的 key 和 value 向量仅依赖于先前的 token。因此，在后续迭代中添加新 token 时，现有 token 的 key 和 value 向量将保持不变

对于第一层，这一概念的验证相对简单：token 的 key 向量是通过将 token 的固定嵌入向量与固定的 wk 参数矩阵相乘来确定的。因此，在后续的迭代中，无论是否引入额外的 token，它都不会发生变化，同样的道理也适用于 value 向量。