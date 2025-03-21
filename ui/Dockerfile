# 使用官方 Python 3.9-slim 作为基础镜像
FROM python:3.12

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目所有文件到镜像中
COPY . .

# 暴露默认 Streamlit 端口（仅作文档用途，实际监听端口由环境变量 PORT 控制）
EXPOSE 8501

# 注意：Cloud Run 会设置环境变量 PORT，我们需要让 Streamlit 监听该端口
# 为了使环境变量正确解析，推荐使用 shell 形式的 CMD
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
