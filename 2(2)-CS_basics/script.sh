#!/usr/bin/env bash
# chmod +x script.sh
set -e

############################################
# 1) Python3 설치 여부 확인
############################################
if ! command -v python3 &> /dev/null; then
    echo "[INFO] Python3 is not installed. Installing Python3..."
    sudo apt-get update -y
    sudo apt-get install -y python3
    echo "[INFO] Python3 installed."
else
    echo "[INFO] Python3 is already installed."
fi


############################################
# 2) pip3 설치 여부 확인
############################################
if ! command -v pip3 &> /dev/null; then
    echo "[INFO] pip3 is not installed. Installing pip3..."
    sudo apt-get update -y
    sudo apt-get install -y python3-pip
    echo "[INFO] pip3 installed."
else
    echo "[INFO] pip3 is already installed."
fi


############################################
# 3) Miniconda 설치 여부 확인
############################################


if [ -x "$HOME/miniconda/bin/conda" ]; then
    echo "[INFO] Miniconda appears to be installed at $HOME/miniconda."

    export PATH="$HOME/miniconda/bin:$PATH"

    if ! command -v conda &> /dev/null; then
        echo "[INFO] Adding Miniconda to PATH."
        export PATH="$HOME/miniconda/bin:$PATH"
    fi
else
    echo "[INFO] Miniconda is not found. Installing Miniconda..."

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh

    bash /tmp/miniconda.sh -b -p "$HOME/miniconda"

    rm /tmp/miniconda.sh

    export PATH="$HOME/miniconda/bin:$PATH"

    echo "[INFO] Miniconda installed successfully."
fi


############################################
# 4) Conda init 하기기
############################################
export PATH="$HOME/miniconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate base


############################################
# 5) Myenv 환경 설치 및 활성화
############################################
if conda env list | grep -q "^myenv "; then
    echo "[INFO] The environment 'myenv' already exists."
else
    echo "[INFO] Creating conda environment 'myenv' with Python 3.9..."
    conda create -n myenv python=3.9 -y
fi

echo "[INFO] Activating 'myenv'..."
source ~/.bashrc
conda init
conda activate myenv


############################################
# 6) mypy 설치
############################################
if ! python -c "import mypy" &> /dev/null; then
    echo "[INFO] Mypy is not found. Installing mypy..."
    conda install -y mypy
else
    echo "[INFO] mypy is already installed."
fi
# echo "[INFO] Installing mypy in 'myenv' environment..."
# conda install -y mypy


############################################
# 7) submission에 있는 python script 실행
############################################
# 7-1) 동적 경로 설정 및 디렉토리 생성, 확인
script_dir=$(dirname "$0")
input_dir="$script_dir/input"
output_dir="$script_dir/output"
submission_dir="$script_dir/submission"

mkdir -p "$input_dir"
mkdir -p "$output_dir"
mkdir -p "$submission_dir"

if [[ ! -d "$input_dir" ]] || [[ -z $(ls -A "$input_dir") ]]; then
    echo "[ERROR] Input directory '$input_dir' does not exist or is empty."
    exit 1
fi

if [[ ! -d "$submission_dir" ]] || [[ -z $(ls -A "$submission_dir"/*.py 2>/dev/null) ]]; then
    echo "[ERROR] No Python files found in '$submission_dir'."
    exit 1
fi



echo "[INFO] Running Python scripts from submission/ folder..."
for file in "$submission_dir"/*.py; do
    filename=$(basename "$file" .py)
    input_file="$input_dir/${filename}_input"
    output_file="$output_dir/${filename}_output"

    if [[ ! -f "$input_file" ]]; then
        echo "[WARNING] Input file '$input_file' does not exist. Skipping $filename."
        continue
    fi

    echo "[INFO] Executing $filename with input=$input_file and output=$output_file"
    python "$file" < "$input_file" > "$output_file"
done


############################################
# 8) mypy 테스트
############################################
echo "Running mypy tests…"
mypy "$submission_dir"/*.py

echo "All tasks are complete."


############################################
# 9) 종료 확인
############################################
echo "[INFO] script.sh has finished all tasks."