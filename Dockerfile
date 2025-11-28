# Docker는 위에서 아래로 차례로 설치하고, Layer cache를 사용해서, 변경이 생긴 다음 레이어부터 다시 설치함(이전까지는 기존 cache 사용)
# 1. 베이스 이미지: PyTorch 대신 가벼운 CUDA devel 이미지 사용 (용량 절약)
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 2. 환경 변수 설정
ENV NVM_DIR="/home/appuser/.nvm" \
    # Poetry가 시스템 경로가 아닌 사용자 경로에 설치되도록 설정
    POETRY_HOME="/home/appuser/.local" \
    PATH="/home/appuser/.local/bin:/home/appuser/.nvm/versions/node/v20.16.0/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    # 컨테이너 내이므로 가상환경 만들지 않음 (단, 사용자 권한 문제로 pip user install 활용 고려)
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true

# 3. 시스템 패키지 설치 (Root)
# root는 집주인, appuser는 세입자
# USER root는 잠시 관리자 모드로 전환 : sudo 등 특정 명령은 관리자만 실행 가능함
USER root
# 타임존 설정 (설치 중 멈춤 방지)
# Docker 빌드는 자동으로 이루어지므로, 질문이 들어오면 대답할 수 없어서 멈춤 -> 묻지 말고 전부 기본값으로 진행하라는 의미
ENV DEBIAN_FRONTEND=noninteractive 
# 라이브러리 최신 버전으로 업데이트 후 설치, 다만 recommends는 제외하고 꼭 필요한 것만 설치
# 그 아래는 필요한 패키지들 나열
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    curl \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python
# 아래에서 2번째는 python3를 python으로 심볼릭 링크(python만 입력해도 3.11 실행)
# 4. 사용자 생성 및 설정
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app
USER appuser

# 5. Node.js & Gemini CLI 설치 (사용자님 요청 유지)
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash && \
    . "$NVM_DIR/nvm.sh" && \
    nvm install 20 && \
    nvm use 20 && \
    npm install -g @google/gemini-cli

# 6. Poetry 설치
RUN curl -sSL https://install.python-poetry.org | python3 -

# 7. 의존성 파일 복사
COPY --chown=appuser:appuser pyproject.toml poetry.lock* ./

# 8. 의존성 설치 (GPU 가속의 핵심!)
# 주의: llama-cpp-python은 빌드 시점에 환경변수가 필요하므로 별도로 빼서 설치하는 것이 안전합니다.
RUN poetry install --no-root --no-interaction && \
    CMAKE_ARGS="-DGGML_CUDA=on" poetry run pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

# 9. 소스 코드 복사
COPY --chown=appuser:appuser . .

# 10. 실행 명령어 : 컨테이너 실행 시(docker run) 자동으로 실행되는 명령어 모음
# poetry run은 컨테이너 내에서 가상환경을 실행하며, poetry를 통해 설치된 라이브러리들을 사용할 수 있게 해주기 때문에, 무조건 실행되어야 함
# 그 후에 추가적인 main.py 등이 수행되도록 설정하여야 함
CMD ["poetry", "run", "python", "main.py"]
