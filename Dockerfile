FROM continuumio/miniconda3:23.10.0-1

WORKDIR /qlib

COPY . .

# 替换Debian bullseye的软件源为阿里云镜像（适配当前系统版本）
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
        sed -i 's/security.debian.org/mirrors.aliyun.com\/debian-security/g' /etc/apt/sources.list


RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean  # 清理缓存，减小镜像体积

RUN conda create --name qlib_env python=3.12 -y
RUN echo "conda activate qlib_env" >> ~/.bashrc
ENV PATH=/opt/conda/envs/qlib_env/bin:$PATH

RUN python -m pip install --upgrade pip

# 安装QLib依赖（调整版本以适配Python 3.12）
# 注意：Python 3.12需要依赖库支持，以下版本经过测试兼容3.12
RUN python -m pip install numpy>=1.26.0  # 1.26+开始支持Python 3.12
RUN python -m pip install pandas>=2.1.4  # 2.1+支持3.12
RUN python -m pip install importlib-metadata>=6.0.0
RUN python -m pip install "cloudpickle<3"  # 保持原版本，兼容QLib
RUN python -m pip install scikit-learn>=1.3.2  # 1.3+支持3.12
# 新增：安装setup.py需要的setuptools-scm
RUN python -m pip install setuptools-scm



# 其他依赖（确保版本兼容3.12）
RUN python -m pip install cython>=0.29.36 packaging tables matplotlib statsmodels>=0.14.0
RUN python -m pip install pybind11>=2.11.1 cvxpy>=1.3.2
RUN python -m pip install ruamel.yaml
RUN python -m pip install pydantic-settings
RUN python -m pip install dill "filelock>=3.16.0" fire gym jupyter lightgbm loguru mlflow nbconvert pyarrow pymongo python-redis-lock pyyaml redis tqdm

# 构建参数：是否稳定版（默认yes）
ARG IS_STABLE="yes"

# 手动指定版本号，绕过setuptools-scm的Git检测
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0

RUN if [ "$IS_STABLE" = "yes" ]; then \
        python -m pip install pyqlib; \
    else \
        python setup.py install; \
    fi
