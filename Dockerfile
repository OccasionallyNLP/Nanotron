FROM huggingface/transformers-pytorch-gpu:latest

# 작업 디렉토리 설정
WORKDIR /workspace

# 현재 로컬 폴더를 Docker 이미지에 복사
COPY . /workspace/Nanotron

# 필요사항 설치
RUN pip install packaging ninja triton "flash-attn>=2.5.0" --no-build-isolation 

# your_project 디렉토리로 이동하여 pip 설치
RUN pip install --upgrade pip && \
    pip install -e /workspace/Nanotron