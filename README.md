# slambook2-remake
视觉SLAM十四讲个人重制

原仓库：https://github.com/gaoxiang12/slambook2

去掉了3rdparty，原则上希望代码能够适配最新的基础库。

## 安装
由于部分基础库需要编译安装，可能会损坏宿主机环境，可以参考下面的容器安装：
```
# 也可以使用最简单的ubuntu镜像，不用带上cuda
docker pull nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# 按需启动容器，其中的端口映射是用来启动远程访问的，可以忽略。
docker run -it --gpus all \
    -v /mnt/data/workspace:/workspace \
    --env="DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v /run/user/${UID}:/run/user/${UID}:rw \
    -p 8888:8888 \
    -p 8022:22 \
    nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

```

进入容器后，安装相应的依赖库。
```
# 按需安装
apt update && \
apt install -y openjdk-17-jdk \
               vim \
               openssh-server \
               cmake \
               g++ \
               git \
               libgoogle-glog-dev \
               libfmt-dev \
               libgl1-mesa-dev \
               libepoxy-dev \
               python3-pip \
               libeigen3-dev \
               libceres-dev \
               libopencv-dev \
               freeglut3-dev \
               qtbase5-dev \
               libmetis-dev \
               libqglviewer-dev-qt5 \
               libcanberra-gtk-module \
               libpcl-dev \
               pcl-tools \
               liboctomap-dev \
               octovis

# 如果需要远程访问，可以使用如下命令：
mkdir /var/run/sshd
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
ssh-keygen -A
echo 'root:YOUR_PASSWORD' | chpasswd
service ssh start
```

由于Ubuntu的软件仓库不包含相关模块，可以使用下面的链接进行下载安装（容器内）：
- https://github.com/strasdat/Sophus
- https://github.com/stevenlovegrove/Pangolin
- https://github.com/RainerKuemmerle/g2o
- ~~https://github.com/rmsalinas/DBow3~~ (compile error)
- https://github.com/dengfeng2/DBow3

如果在安装上述库时，不想破坏本机环境，可以先将库安装在一个特定目录下，比如：
```shell
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/.source-install
make -j8 && make install
```
然后在运行本仓库代码时，需要添加cmake选项：
```shell
cmake .. -DCMAKE_PREFIX_PATH=~/.source-install
```

