# Python
## Features
### `lambda`  
匿名函数,主要目的是减少代码冗余,可读性更强,程序更加简洁  
#### 语法
`lambda argument_list:expersion`   
- 参数列表与python中函数一致
- 表达式中出现的参数需要在参数列表中有定义
- 表达式只能是单行的
#### Examples
1. 直接赋给一个变量，然后再像一般函数那样调用
```
c=lambda x,y,z:x*y*z
c(2,3,4)

24
```
2. 也可以在函数后面直接传递实参
```
(lambda x:x**2)(3)
9
```
3. 将lambda函数作为参数传递给其他函数比如说结合map、filter、sorted、reduce等一些Python内置函数使用
```
fliter(lambda x:x%3==0,[1,2,3,4,5,6])

[3,6]


squares = map(lambda x:x**2,range(5))
print(lsit(squares))
[0,1,4,9,16]

# 与sort结合
a=[('b',3),('a',2),('d',4),('c',1)]
sorted(a,key=lambda x:x[0])
[('a',2),('b',3),('c',1),('d',4)]
```
