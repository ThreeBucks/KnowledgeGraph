# OP Fusion
## 背景
优化data movement是高性能计算永恒的主题.无论计算机架构怎么变, bottleneck总是在数据IO上(而非计算).**算子融合主要目的是减少内存访问(让数据更快的被重利用).**  
一般而言, 被合并的算子之间需要有数据依赖,才有可能起到减少内存访问的效果. 但是有些情况对于毫无关系的两个算子,合并他们可能也会有好处, 比如把两个GPU kernel合并成一个, 减小launch的开销. 或者合并两个循环, 减小循环本身带来的开销.  
算子合并的输入是一个DAG图, 表示了不同操作之间的数据流, 依赖关系. 这个图一般仅仅表示数据流, 而不包括控制流(对于有控制流的情况变成了另外一个难题). 输出是一些合并了的kernel, 比如可能是C++代码(CPU上)或者PTX(GPU上), 或者其他中间形式. 这个kernel除了fusion还会包含其他重要的优化如tiling. 这个kernel代码可能是100%编译器生成的, 可能是自动生成+人工优化的算子库结合, 也可能是100%人工优化的算子库.
## 算子类型
### 访存密集型
从合并难度的角度, 算子又被分为pointwise(elementwise)和reduction. pointwise-pointwise, pointwise-reduction之间的融合比较简单, 很多编译器都会做, 示例:
```
A = relu(B+C)
```
直接解释执行需要先计算`B+C`(矩阵和), 然后存储中间结果到一个新的矩阵, 然后对新的矩阵计算`relu`得到`A`. 这里的一大开销就是保存中间结果到内存, 造成额外的内存访问开销.  
如果使用一个能够自动融合的编译器, 生成的代码大概是:
```
for (int i = 0; i < NI; i++) {
    A[i] = relu(B[i] + C[i])
}
```
这样就不需要保存中间结果到内存(中间结果在寄存器). 值得一提的是, 并非所有的pointwise, reduction算子都是可以合并的, 示例, row-wise softmax:
```
A = sum(B, axis=1)
C = B / A[:, None] 
```
用循环的形式写出来:
```
for (int i = 0; i < NI; i++) {
    for (int j = 0; j < NJ; j++) {
        A[i] += B[i,j];  // S1
    }
}

for (int i = 0; i < NI; i++) {
    for (int j = 0; j < NJ; j++) {
        C[i,j] = B[i,j] / A[i];  // S2
    }
}
```
这两个算子就无法被合并进同一个最内层循环, 不然数据依赖就会被打破, 比如一下合并就是不合法的代码(从S1到S2关于`A`的RAW依赖被打破, 出现了WAR):
```
for (int i = 0; i < NI; i++) {
    for (int j = 0; j < NJ; j++) {
        A[i] += B[i,j];   // S1
        C[i,j] = B[i,j] / A[i];  // S2  
    }
}
```
但以下依赖是合法的(保持了数据依赖):
```
for (int i = 0; i < NI; i++) {
    for (int j = 0; j < NJ; j++) {
        A[i] += B[i,j];  // S1
    }

    for (int j = 0; j < NJ; j++) {
        C[i,j] = B[i,j] / A[i];  // S2
    }
}
```
合并后`A[i]`可以用一个scalar变量代替（scalar replacement）。好在在深度学习的应用场景下数据依赖关系往往比较简单，而且也没有多少控制依赖，但是对于科学计算程序，编译器则需要考虑到任意的依赖，合并就必须考虑到合法性。
### 计算密集型
比如有matmul, convolution. 合并matmul等计算密集型和pointwise则比较困难, 原因是编译器很难自动生成能match手工优化的matmul算子.  
在很多应用场景中多个连续的matmul也是常见的模式, 这种情况想要在fuse的同时还使得多个matmul的部分保持算子库般的性能更加困难, 比如[Flash Attention](./flash_attention.md)介绍了一种手动的合并matmul+softmax+matmul的办法(非编译器自动合并)



