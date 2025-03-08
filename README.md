# U-net

使用了Encoder-Decoder 结构

Encoder过程通道逐渐增加，但是长宽减少，实现了对特征的有效提取，Decoder进行上采样过程，使得图像逐渐恢复原样

其中关键是使用了矩阵跳跃拼接的方式，**feature map 的维度没有变化，但每个维度都包含了更多特征，相比于复制的相加，更加高效**，下采样过程会使得特征性增强，但细节会减少，这刚好弥补了这一点，使得上采样过程保留更多细节

在拼接时使用了双线性插值进行的拼接，使得图像像素更加真实

但参数任然过大，缺乏上下文理解，可以尝试与tansformer共同使用
