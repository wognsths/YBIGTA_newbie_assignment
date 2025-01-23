#!/bin/bash

# miniconda가 존재하지 않을 경우 설치
if ! command -v conda &> /dev/null && [ ! -d "$HOME/miniconda" ]; then
    echo "[INFO] Miniconda 설치 중..."
    # Miniconda 설치 명령어 (Linux 기준)
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    # .bashrc에 경로 추가 (터미널 새로 열었을 때 적용)
    echo '[INFO] export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    echo "[INFO] Miniconda 설치 완료"
else
    echo "[INFO] Miniconda가 이미 설치되어 있습니다."
fi

# Conda 환경 생성 및 활성화
source $HOME/miniconda/etc/profile.d/conda.sh

if ! conda env list | grep "myenv" &> /dev/null; then
    echo "[INFO] Conda 가상환경 'myenv' 생성 중..."
    conda create -n myenv python=3.8 -y
fi

echo "Conda 가상환경 활성화 중..."
conda activate myenv

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "[INFO] 가상환경 활성화: 성공"
else
    echo "[INFO] 가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
echo "[INFO] 필요한 패키지 설치 중..."
# mypy 설치 여부 확인 및 설치
if ! command -v mypy &> /dev/null; then
    echo "[INFO] mypy가 설치되어 있지 않습니다. 설치 중..."
    conda install -y -c conda-forge mypy
else
    echo "[INFO] mypy가 이미 설치되어 있습니다."
fi

# Submission 폴더 파일 실행
cd submission || { echo "[INFO] submission 디렉토리로 이동 실패"; exit 1; }

# Python 파일 실행
for file in *.py; do
    problem_name=$(basename "$file" .py)  # 파일명에서 문제명 추출
    input_dir="../input/${problem_name}_input"  # 입력 디렉토리 경로
    output_dir="../output/${problem_name}_output"  # 출력 디렉토리 경로

    # 문제에 대한 입력을 받고 출력 파일을 생성
    python "$file" < "$input_dir" > "$output_dir" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "[INFO] $file 실행 완료, 출력 파일: $output_dir"
    else
        echo "[INFO] $file 실행 중 에러 발생"
    fi
done

# mypy 테스트 실행
echo "[INFO] mypy 테스트 실행 중..."
mypy *.py >/dev/null 2>&1

# 가상환경 비활성화
echo "[INFO] 가상환경 비활성화 중..."
conda deactivate
