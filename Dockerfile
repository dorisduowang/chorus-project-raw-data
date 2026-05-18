FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for unstructured
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    pandoc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install numpy<2 first, then unstructured
RUN pip install --no-cache-dir "numpy<2" && \
    pip install --no-cache-dir "unstructured[pdf,docx,pptx]" python-magic

# Install ipykernel for VS Code notebook support
RUN pip install --no-cache-dir ipykernel && \
    python -m ipykernel install --user --name python3 --display-name "Python 3.11"

# Install RAG dependencies
RUN pip install --no-cache-dir \
    sentence-transformers \
    faiss-cpu \
    rank_bm25 \
    openai \
    anthropic \
    fastmcp \
    slack-bolt \
    aiohttp

# Copy project files
COPY . .

# Expose Jupyter and RAG server ports
EXPOSE 8888 8765

# Run Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
