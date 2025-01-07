import os
import requests
from tqdm import tqdm

def download_file(url: str, save_path: str):
    """파일 다운로드 함수"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as file, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def download_dialogsum(data_dir: str):
    """DialogSum 데이터셋 다운로드"""
    # 데이터 디렉토리 생성
    os.makedirs(data_dir, exist_ok=True)
    
    # DialogSum 데이터셋 URL (올바른 URL로 수정)
    base_url = "https://raw.githubusercontent.com/cylnlp/dialogsum/main/DialogSum_Data"
    files = {
        'train.json': f"{base_url}/dialogsum.train.jsonl",
        'val.json': f"{base_url}/dialogsum.dev.jsonl"
    }
    
    # 각 파일 다운로드
    for filename, url in files.items():
        save_path = os.path.join(data_dir, filename)
        if not os.path.exists(save_path):
            print(f"Downloading {filename}...")
            try:
                download_file(url, save_path)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        else:
            print(f"{filename} already exists.")
