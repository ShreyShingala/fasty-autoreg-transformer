FROM python:3.11-slim
ENV PIP_DEFAULT_TIMEOUT=120 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir triton==3.1.0 transformers==4.51.3 safetensors==0.5.3 tokenizers==0.21.1 numpy==2.1.3
WORKDIR /work
ENV PYTHONPATH=/work/engine
