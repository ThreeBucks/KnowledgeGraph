# Docker
## 多阶段构建
Docker 17.05 版本开始支持, 只需要一个Dockerfile就可以解决编译环境和运行环境隔离的问题  
### Features
- 在一个Dockerfile里使用多个From语句, 从而创建多个Stage,每个阶段间独立
- 可以通过`COPY --from=Builder xxx xxx`来获取其他阶段的文件, 这里的`Builder`是指定的阶段名称, 也可以指定阶段的index
- 只有最后一个stage的内容会保留在镜像中
- 支持调试(使构建停在某个阶段): `docker build --target build1 .`

### Example

```
# 第一阶段——编译
FROM openjdk:8u171-jdk-alpine3.8 as builder # 自带编译工具
ADD . /app
WORKDIR /app
RUN ... 省略编译和清理工作...
 
 
# 现在，JAR 已经出炉。JDK 不再需要，所以不能留在镜像中。
# 所以我们开启第二阶段——运行，并扔掉第一阶段的所有文件（包括编译工具）
# 第二阶段——运行
FROM openjdk:8u181-jre-alpine3.8 as environment # 只带运行时环境
 
# 目前，编译工具等上一阶段的东西已经被我们抛下。目前的镜像中只有运行时环境，
# 我们需要把上一阶段的结果拿来，其它不要。
COPY --from=0 /final.jar .
 
# 好了，现在镜像只有必要的运行时和 JAR 了。
ENTRYPOINT java -jar /final.jar

```
