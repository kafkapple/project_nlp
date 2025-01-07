import json
import os
from src.logger.logger import get_logger

def preprocess_dialogsum(data_dir):
    """DialogSum 데이터 전처리 및 outputs 폴더에 메트릭 결과 저장"""
    logger = get_logger(__name__)
    
    # outputs 디렉토리 생성
    outputs_dir = os.path.join(os.path.dirname(data_dir), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    metrics = {}
    
    for split in ['train', 'val']:
        input_file = os.path.join(data_dir, f'{split}.json')
        
        try:
            # 전체 파일을 한번에 JSON으로 읽기
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 데이터 통계 수집
            metrics[split] = {
                'total_examples': len(data),
                'avg_dialogue_length': sum(len(d['dialogue'].split()) for d in data) / len(data),
                'avg_summary_length': sum(len(d['summary'].split()) for d in data) / len(data)
            }
            
            # 처리된 데이터 저장 - 원본 위치에 저장
            processed_file = os.path.join(data_dir, f'{split}.json')
            with open(processed_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Error processing {split} file: {e}")
            raise
    
    # 메트릭 결과를 outputs 폴더에 저장
    metrics_file = os.path.join(outputs_dir, 'data_metrics.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Data preprocessing completed. Metrics saved to {metrics_file}")
