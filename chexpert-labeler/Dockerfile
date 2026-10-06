FROM python:3.7-slim

# Install system dependencies: build tools, Java runtime for NegBio, and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    default-jre \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/chexpert-labeler

# Clone NegBio dependency
RUN git clone https://github.com/ncbi-nlp/NegBio.git
ENV PYTHONPATH=/app/chexpert-labeler/NegBio

# Install Python packages via pip (bioc==1.0 + future for NegBio BioCAnnotation compatibility)
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    nltk \
    tqdm \
    networkx==2.3 \
    bioc==1.0 \
    future \
    pystanforddependencies \
    pathlib2 \
    ply \
    bllipparser==2016.9.11

# Copy chexpert-labeler source code
COPY . .

# Download NLTK datasets and bllipparser GENIA model
RUN python -m nltk.downloader universal_tagset punkt wordnet omw-1.4
RUN python -c "from bllipparser import RerankingParser; RerankingParser.fetch_and_load('GENIA+PubMed')"

# Warm-up / cache CoreNLP on sample reports
RUN python label.py --reports_path sample_reports.csv --output_path labeled_reports.csv --verbose

CMD ["python", "label.py", "--help"]
