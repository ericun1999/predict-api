FROM python:3.13.0-slim
WORKDIR /app

# 安装系统依赖并清理缓存
RUN apt-get update && apt-get install -y \
    gcc \
    libpng-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip  # 清理pip缓存以减小镜像体积

# 复制应用代码
COPY . .

# 暴露端口（仅为文档说明，实际由PORT环境变量决定）
EXPOSE 8080

# 使用Shell格式执行命令，确保环境变量正确解析
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app
