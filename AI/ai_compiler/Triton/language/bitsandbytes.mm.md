# BitsandBytes
## Reference
- [github](https://github.com/TimDettmers/bitsandbytes)

## 介绍
是一个低精度(主要是8bit)的CUDA 核函数库,支持训练和推理
### Feature
- 支持混合精度的8bit 矩阵乘
- LLM.int8() 推理
- 8-bit优化器: Adam, AdamW, RMSProp, LARS, LAMB, Lion(节省75%显存)
- 8-bit[量化](../../../量化/intro.mm.md)

