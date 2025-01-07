import json
import os

def preprocess_dialogsum(data_dir: str):
    """DialogSum 데이터 전처리"""
    files = ['train.json', 'val.json']
    
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print(f"Warning: {filename} not found in {data_dir}")
            continue
            
        # JSONL 파일을 JSON 배열로 변환
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        # 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
