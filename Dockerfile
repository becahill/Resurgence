FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  build-essential \
  curl \
  pkg-config \
  libssl-dev \
  ca-certificates \
  git \
  && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /workspace
COPY pyproject.toml README.md /workspace/
COPY resurgence_core /workspace/resurgence_core
COPY resurgence_py /workspace/resurgence_py
COPY tests /workspace/tests

RUN pip install --upgrade pip \
  && pip install -e .[dev] \
  && maturin develop --release

CMD ["python", "-m", "resurgence_py.main", "--tickers", "SPY,QQQ,IWM"]
