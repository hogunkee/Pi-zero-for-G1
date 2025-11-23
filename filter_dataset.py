#!/usr/bin/env python3
"""
LeRobot 데이터셋에서 state modality를 필터링하고 
parquet, modality.json, norm_stats.json을 수정하는 스크립트
"""

import json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


def filter_observation_state(state_value):
    """observation.state에서 앞 15개 dof 제거 (index 15부터만 유지)"""
    if isinstance(state_value, (list, np.ndarray)):
        return state_value[15:]
    return state_value


def filter_parquet_files(dataset_path):
    """데이터셋 내 모든 parquet 파일의 observation.state 필터링"""
    dataset_path = Path(dataset_path)
    parquet_files = list(dataset_path.rglob("*.parquet"))
    
    print(f"\n📦 Parquet 파일 처리 중...")
    print(f"  발견된 파일: {len(parquet_files)}개")
    
    for parquet_file in tqdm(parquet_files, desc="Filtering parquet files"):
        try:
            df = pd.read_parquet(parquet_file)
            if "observation.state" in df.columns:
                df["observation.state"] = df["observation.state"].apply(filter_observation_state)
                df.to_parquet(parquet_file)
        except Exception as e:
            print(f"  ❌ 오류 ({parquet_file}): {e}")
    
    print(f"  ✓ 모든 parquet 파일 처리 완료")


def filter_and_copy_dataset(source_dataset_path, dest_dataset_path):
    """데이터셋을 복제하고 state를 필터링하며 모든 파일을 수정"""
    
    source_path = Path(source_dataset_path)
    dest_path = Path(dest_dataset_path)
    
    print("=" * 70)
    print("LeRobot 데이터셋 필터링 시작")
    print("=" * 70)
    
    if dest_path.exists():
        print(f"\n⚠️  '{dest_dataset_path}'가 이미 존재합니다. 삭제하고 진행합니다.")
        shutil.rmtree(dest_path)
    
    print(f"\n📂 데이터셋 복제 중...")
    shutil.copytree(source_path, dest_path)
    print(f"✓ 데이터셋 복제 완료")
    
    # Parquet 파일들 필터링
    filter_parquet_files(dest_path)
    
    # modality.json 수정
    modality_path = dest_path / "meta" /  "modality.json"
    print(f"\n📝 modality.json 수정 중...")
    
    with open(modality_path, 'r') as f:
        modality = json.load(f)
    
    new_state = {
        "left_arm": {"start": 0, "end": 7},
        "right_arm": {"start": 7, "end": 14},
        "left_hand": {"start": 14, "end": 21},
        "right_hand": {"start": 21, "end": 28}
    }
    modality["state"] = new_state
    
    with open(modality_path, 'w') as f:
        json.dump(modality, f, indent=2)
    
    print(f"  ✓ modality.json 수정 완료")
    
    # norm_stats.json 수정
    norm_stats_path = dest_path / "g1" / "norm_stats.json"
    print(f"\n📊 norm_stats.json 수정 중...")
    
    if norm_stats_path.exists():
        with open(norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
        
        for stat_key in ["mean", "std", "q01", "q99"]:
            norm_stats["norm_stats"]["state"][stat_key] = \
                norm_stats["norm_stats"]["state"][stat_key][15:]
        
        with open(norm_stats_path, 'w') as f:
            json.dump(norm_stats, f, indent=2)
        
        print(f"  ✓ norm_stats.json 수정 완료")
    
    print(f"\n✅ 모든 작업 완료!\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LeRobot 데이터셋 필터링")
    parser.add_argument("source", help="원본 데이터셋 경로")
    parser.add_argument("dest", help="새로운 데이터셋 경로")
    
    args = parser.parse_args()
    filter_and_copy_dataset(args.source, args.dest)
