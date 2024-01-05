# Compute Graph
## 类别
PyTorch框架2.0版本之前支持两套计算图模式，分别通过Torch FX（torch.fx）和TorchScript（torch.jit.script）使用,其中:
- TorchScript出现时间更久，功能也更稳定，工业界也有使用该特性进行部署的案例
- 而功能更新、适配性更广的Torch FX是PyTorch 1.10以后才加入stable，且在后续的2.0版本中作为重要的特性被引入。

Pytorch 2.0引入了dynamo, 是第三套计算图模式，该模式功能更全，能覆盖的场景也更广. 后续会作为torch compiler的主要演进方向
## 总结
1. TorchScript能支持模型范围较广，且使用容易，但无法支持动态分支，且在某些模型上仍然会报错，无法成功trace
2. TorchFX抽象性好，完成后速度快、模型改写方便，但使用symbolic trace除了不支持动态分支，也不支持使用展开输入的模型，能覆盖模型范围最小
3. TorchDynamo较新，从设计思路上看能够支持动态分支，但仍然有部分功能不完善，但好处是能导出到TorchFX的Graph，也能方便的改写模型
4. 目前没有一套方案能完美支持从PyTorch模型中拿到用IR表示的图，且基于PyTorch进行模型优化的各家工具均需要自己定义一套IR
## 名词解释
- Trace: 符号追踪器, 对模型进行符号推理，给定输入，对模型进行forward，拿到模型推理对数据流，构建计算图
- CodeGen: 代码生成, 对计算图执行推理时，会自动生成的python代码
- IR: 中间表示(intermediate representation), 对模型进行变换的核心部分，在对模型进行符号追踪时已经基于定义好的IR构建了计算图
## Torch Script
### 简介
TorchScript发布较早，**能够实现模型序列化、编译等，更偏向模型部署使用**，例如导出可在C++后端使用的PyTorch模型、对模型进行变换、优化等
### Trace
使用jit.trace进行追踪，需要给定数据（指定shape、dtype等），返回一个torch.jit.ScriptModule，如果tracer的输入是一个function，则会返回一个torch.jit.ScriptMethod：
```

import torch
fake_inputs = (torch.randn(1,3,32,32), torch.randn(1,3,32,32))
traced_mod = torch.jit.trace(model, fake_inputs)
def func(inputs):
    ...
traced_fun = torch.jit.trace(func, inputs)
```
详细的参数注释见：[torch.jit.trace](https://github.com/pytorch/pytorch/blob/main/torch/jit/_trace.py)

对一些有控制流、变长for loop的函数，可以用装饰器@torch.jit.script去包装

### CodeGen
分级生成，粒度较粗：
```
>>> traced_mod.code
def forward(self,
    x: Tensor) -> Tensor:
  fc = self.fc
  avgpool = self.avgpool
  layer4 = self.layer4
  layer3 = self.layer3
  layer2 = self.layer2
  layer1 = self.layer1
  maxpool = self.maxpool
  relu = self.relu
  bn1 = self.bn1
  conv1 = self.conv1
  _0 = (relu).forward((bn1).forward((conv1).forward(x, ), ), )
  _1 = (layer1).forward((maxpool).forward(_0, ), )
  _2 = (layer3).forward((layer2).forward(_1, ), )
  _3 = (avgpool).forward((layer4).forward(_2, ), )
  input = torch.flatten(_3, 1)
  return (fc).forward(input, )
```
可以看到从attributes中还包括了 forward，将递归的生成
### IR
TorchScript定义的IR略复杂，大致有如下：

Modules：对应nn.Module
Parameters：对应nn.Module里的parameters
Method：包括FunctionSchema方法描述，graph实际计算图
FunctionSchema：描述参数与返回类型
Graph：定义function的具体实现，包括了Nodes，Blocks，Values
Nodes：一个指令，如一次卷积、一次矩阵乘
Block：针对控制语句if, loop + list of nodes
Value：
with：
对traced_mod（torch.jit.ScriptModule）使用graph方法，可以看到IR：
```
>>> traced_mod.graph
graph(%self.1 : __torch__.torchvision.models.resnet.ResNet,
      %x.1 : Float(1, 3, 224, 224,
      strides=[150528, 50176, 224, 1],
      requires_grad=0, device=cpu)):
  %fc : __torch__.torch.nn.modules.linear.Linear = \
prim::GetAttr[name="fc"](%self.1)
  %avgpool : __torch__.torch.nn.modules.pooling.\
AdaptiveAvgPool2d = prim::GetAttr[name="avgpool"](%self.1)
  %layer4 : __torch__.torch.nn.modules.container.\
___torch_mangle_58.Sequential = prim::GetAttr[name="layer4"](%self.1)
  %layer3 : __torch__.torch.nn.modules.container.___torch_mangle_42.\
Sequential = prim::GetAttr[name="layer3"](%self.1)
  %layer2 : __torch__.torch.nn.modules.container.___torch_mangle_26.\
Sequential = prim::GetAttr[name="layer2"](%self.1)
  %layer1 : __torch__.torch.nn.modules.container.\
Sequential = prim::GetAttr[name="layer1"](%self.1)
  %maxpool : __torch__.torch.nn.modules.pooling.\
MaxPool2d = prim::GetAttr[name="maxpool"](%self.1)
  ...
```
## Torch FX
### 简介
torch.fx在1.9中发布，1.10中发布了stable版本，主要功能就是**实现对PyTorch nn.Module的变换**
### Trace
使用symbolic_trace进行追踪，即输入是一个默认的假输入（Proxies），返回一个torch.fx.GraphModule：
```
from torch.fx import symbolic_trace
symbolic_traced = symbolic_trace(model)
```
详细的参数注释见：[symbolic_trace](https://github.com/pytorch/pytorch/blob/main/torch/fx/_symbolic_trace.py)
### IR
一个fx.GraphModule包括了一系列的torch.fx.Node，每一个Node都具有特定类别，而torch.fx把所有的IR Node抽象成了六个类别，每个类别都有 name（表示对哪个名称的tensor执行该IR）、target（具体执行的函数、方法等）、args and kwargs（target的输入），不同类别的IR会有不同的name、target和args kwargs，下面是六种IR的解释：

- placeholder：表示计算图输入，可以理解为 forward 中的输入参数
- get_attr：获取一个参数，比如在 forward 中使用 self.attribute，这个操作会转为该IR
- call_function：对调用一些非torch的函数（通常自定义的函数），会把这个调用函数的操作转为该IR
- call_module：对调用nn.Module的forward，会递归的转成该IR
- call_method：对调用torch functions，如torch.min，torch.exp等，会转成该IR
- output：计算图的输出
对symbolic_traced（torch.fx.GraphModule）使用graph.print_tabular()方法，可以看到详细的IR类型即参数：
```
>>> symbolic_traced.print_tabular()
opcode       name           target         args       kwargs
-----------  -------------  -------------  ---------  ------
placeholder  x              x              ()         {}
call_module  conv1          conv1          (x,)       {}
call_module  bn1            bn1            (conv1,)   {}
call_module  relu           relu           (bn1,)     {}
call_module  maxpool        maxpool        (relu,)    {}
call_module  layer1_0_conv  layer1.0.conv1 (maxpool,) {}
...
call_module  fc             fc             (flatten,) {}
output       output         output         (fc,)      {}
```
### CodeGen
生成代码粒度很细：
```
>>> symbolic_traced.code
def forward(self, x : torch.Tensor) -> torch.Tensor:
    conv1 = self.conv1(x);  x = None
    bn1 = self.bn1(conv1);  conv1 = None
    relu = self.relu(bn1);  bn1 = None
    maxpool = self.maxpool(relu);  relu = None
    layer1_0_conv1 = getattr(self.layer1, "0").conv1(maxpool)
    layer1_0_bn1 = getattr(self.layer1, "0").bn1(layer1_0_conv1);  layer1_0_conv1 = None
    layer1_0_relu = getattr(self.layer1, "0").relu(layer1_0_bn1);  layer1_0_bn1 = None
    layer1_0_conv2 = getattr(self.layer1, "0").conv2(layer1_0_relu);  layer1_0_relu = None
    layer1_0_bn2 = getattr(self.layer1, "0").bn2(layer1_0_conv2);  layer1_0_conv2 = None
    add = layer1_0_bn2 + maxpool;  layer1_0_bn2 = maxpool = None
    layer1_0_relu_1 = getattr(self.layer1, "0").relu(add);  add = None
    ...
    flatten = torch.flatten(avgpool, 1);  avgpool = None
    fc = self.fc(flatten);  flatten = None
    return fc
```


## 对比
|计算图|优点|缺点|
|----|----|----|
|Torch Script|更成熟，使用上对各种模型结构、方法的支持更稳定|1. 更多的是针对不同平台、硬件部署使用; 2. IR定义较为复杂，对一个ScriptModule进行算子替换也比较麻烦; 3. 对一些控制流（if confidition）、动态循环（for loop）均没有支持，仅在第一次trace的时候固定计算图|
|Torch FX|1. 主要面向模型优化和变换，且功能更新，和最近PyTorch开始主推的Dynamo和Inductor能够相互支持; 2. 对IR的抽象好|模型结构支持还有限，且tracer是使用Proxies，碰到一些需要根据动态shape去临时生成变量的操作就比较困难（如torch.arange这种）|
|Dynamo||||
