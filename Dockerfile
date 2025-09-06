FROM python:3.10-slim
WORKDIR /app

# Java 11 for PySpark
RUN apt-get update && apt-get install -y \
    curl jq unzip libpq-dev gcc procps wget git \
 && wget https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.25%2B9/OpenJDK11U-jdk_x64_linux_hotspot_11.0.25_9.tar.gz \
 && mkdir -p /opt/jdk-11 \
 && tar -xzf OpenJDK11U-jdk_x64_linux_hotspot_11.0.25_9.tar.gz -C /opt/jdk-11 --strip-components=1 \
 && rm OpenJDK11U-jdk_x64_linux_hotspot_11.0.25_9.tar.gz \
 && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/opt/jdk-11
ENV PATH=$JAVA_HOME/bin:$PATH

# Root
COPY requirements.txt ngrok.yml start.sh ./

# Borrower Chatbot
COPY borrower_chatbot/ /app/borrower_chatbot/

# Officer Chatbot
COPY officer_chatbot/ /app/officer_chatbot/

# Airflow
COPY airflow /app/airflow

# RAG support
COPY info/keywords.json /app/info/
COPY rag_vectorstore/index.faiss rag_vectorstore/index.pkl /app/rag_vectorstore/

# Model artifacts
COPY model_deployment/ /app/model_deployment/

# MLflow run path
COPY mlruns/787990574769484173/37d2cbaaf3c94eceb7e1a1bc2f09d9ea \
     /app/mlruns/787990574769484173/37d2cbaaf3c94eceb7e1a1bc2f09d9ea

RUN pip install --upgrade pip

# with cache
RUN pip install --no-cache-dir -r requirements.txt
# without cache
# RUN pip install -r requirements.txt

# ngrok agent
RUN curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
 && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | tee /etc/apt/sources.list.d/ngrok.list \
 && apt-get update && apt-get install -y ngrok && rm -rf /var/lib/apt/lists/*

# Clean up Python cache files and logs
RUN find /app -name "*.pyc" -delete && \
    find /app -name "*.pyo" -delete && \
    find /app -name "*.pyd" -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /app -name "*.log" -delete

RUN chmod +x /app/start.sh
EXPOSE 5000 5001 4040

ENV MLFLOW_TRACKING_DIR=/app/mlruns
CMD ["./start.sh"]
