FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library from source (libta-lib-dev not in Debian repos)
RUN curl -L https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
    | tar xz \
    && cd ta-lib-0.6.4 \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.6.4

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/{raw,processed,snapshots} models reports logs

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "-m", "src.ops.daily_job"]
