"""
Phase Smoothing Script for LeRobot Dataset
Binary phase 값(0 or 1)을 continuous하게 smoothing합니다.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from typing import Literal


def sigmoid_smooth(x, center, width):
    """Sigmoid 함수를 사용한 부드러운 전환"""
    return 1.0 / (1.0 + np.exp(-(x - center) / width))

def linear_smooth(x, start, end, start_val, end_val):
    """선형 보간"""
    if end <= start:
        return np.full_like(x, start_val, dtype=np.float64)
    alpha = np.clip((x - start) / (end - start), 0, 1)
    return start_val + alpha * (end_val - start_val)
    
def shift_phase_transitions(phase_values, shift_size):
    """
    Phase 값들을 오른쪽으로 shift합니다.

    Args:
        phase_values: Phase 값 배열
        shift_size: 오른쪽으로 shift할 크기

    Returns:
        Shifted phase 값 배열
    """
    phase = np.array(phase_values, dtype=np.float64)

    if shift_size <= 0:
        return phase

    # shift_size가 배열 길이보다 크면 전체를 첫 값으로 채움
    if shift_size >= len(phase):
        return np.full_like(phase, phase[0], dtype=np.float64)

    # 오른쪽으로 shift: 왼쪽은 첫 값으로 채우고, 오른쪽은 원래 값들을 이동
    shifted = np.empty_like(phase, dtype=np.float64)
    shifted[:shift_size] = phase[0]  # 왼쪽에 생기는 공간은 첫 값으로 채움
    shifted[shift_size:] = phase[:-shift_size]  # 나머지는 원래 값을 shift

    return shifted

def smooth_phase_transitions(phase_values, window_size=10, method='sigmoid'):
    """
    Binary phase 값의 전환을 부드럽게 만듭니다.
    
    Args:
        phase_values: Binary phase 값 배열 (0 or 1)
        window_size: Smoothing 적용할 윈도우 크기 (양쪽으로 적용)
        method: 'sigmoid' 또는 'linear'
    
    Returns:
        Smoothed phase 값 (0~1 사이의 float)
    """
    phase = np.array(phase_values, dtype=np.float64)
    smoothed = phase.copy()
    
    # 변화점 찾기 (0→1 또는 1→0)
    diff = np.diff(phase)
    transitions = np.where(diff != 0)[0]  # 변화가 일어난 인덱스
    
    for trans_idx in transitions:
        # 전환 타입 확인
        from_val = phase[trans_idx]
        to_val = phase[trans_idx + 1]

        if from_val < to_val:  # 0 → 1
            start_idx = max(0, trans_idx - window_size)
            end_idx = min(len(phase), trans_idx + 1)
        else:  # 1 → 0
            start_idx = max(0, trans_idx)
            end_idx = min(len(phase), trans_idx + 1 + window_size)
        
        # Smoothing 적용
        if method.split('-')[0] == 'sigmoid':
            # Sigmoid 중심을 transition point로
            center = (start_idx + end_idx)/2
            width = (end_idx - start_idx) / 12.0
            #center = trans_idx + 0.5
            #width = window_size / 6.0  # Sigmoid의 기울기 조절
            
            for i in range(start_idx, end_idx):
                if from_val < to_val:  # 0 → 1
                    smoothed[i] = sigmoid_smooth(i, center, width)
                else:  # 1 → 0
                    smoothed[i] = 1.0 - sigmoid_smooth(i, center, width)
        
        elif method.split('-')[0] == 'linear':
            # 선형 보간
            for i in range(start_idx, end_idx):
                smoothed[i] = linear_smooth(i, start_idx, end_idx - 1, from_val, to_val)
    
    return smoothed

def process_parquet_file(src_path: Path, dst_path: Path, phase_col_idx: int, shift_size: int,
                         window_size: int, method: str, dry_run: bool = False):
    """
    Parquet 파일을 읽어서 phase 컬럼을 smoothing하고 저장
    
    Args:
        src_path: 원본 parquet 파일 경로
        dst_path: 저장할 parquet 파일 경로
        phase_col_idx: Phase 컬럼의 인덱스 (action.phase의 start 값)
        window_size: Smoothing window 크기
        method: Smoothing 방법
        dry_run: True면 저장하지 않고 통계만 출력
    """
    # Parquet 파일 읽기
    table = pq.read_table(src_path)
    df = table.to_pandas()
    
    # Action 컬럼 찾기
    action_col = None
    for col in df.columns:
        if 'action' in col.lower() and col.lower() != 'next.action':
            action_col = col
            break
    
    if action_col is None:
        print(f"⚠️  {src_path.name}: action 컬럼을 찾을 수 없습니다.")
        return None
    
    # Action 데이터 추출 (보통 array 형태)
    action_data = df[action_col].values
    
    # Phase 값 추출
    if isinstance(action_data[0], np.ndarray):
        # Array 형태인 경우
        phase_values = np.array([row[phase_col_idx] for row in action_data])
    else:
        print(f"⚠️  {src_path.name}: action 컬럼 형식을 인식할 수 없습니다.")
        return None
    
    # 변화점 개수 확인
    diff = np.diff(phase_values)
    num_transitions = np.sum(diff != 0)
    
    if num_transitions == 0:
        print(f"⏭️  {src_path.name}: 변화점이 없습니다 (smoothing 불필요)")
        return {
            'file': src_path.name,
            'transitions': 0,
            'original_unique': len(np.unique(phase_values)),
            'smoothed_unique': len(np.unique(phase_values)),
            'smoothed': False
        }
    
    if df['task_index'][0] in [5,6,7,8,9,10,11,12,13,14,15]:
        # Phase shift 적용
        phase_values = shift_phase_transitions(phase_values, shift_size)
    
    # Phase smoothing 적용
    smoothed_phase = smooth_phase_transitions(phase_values, window_size, method)
    
    # Action 데이터에 다시 적용
    modified_action = []
    for i, row in enumerate(action_data):
        new_row = row.copy()
        new_row[phase_col_idx] = smoothed_phase[i]
        modified_action.append(new_row)
    
    df[action_col] = modified_action
    
    # 통계
    stats = {
        'file': src_path.name,
        'transitions': num_transitions,
        'original_unique': len(np.unique(phase_values)),
        'smoothed_unique': len(np.unique(smoothed_phase)),
        'original_range': (phase_values.min(), phase_values.max()),
        'smoothed_range': (smoothed_phase.min(), smoothed_phase.max()),
        'smoothed': True
    }
    
    # 저장
    if not dry_run:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, dst_path)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="LeRobot 데이터셋의 binary phase 값을 smoothing")
    parser.add_argument("root", type=str, help="데이터셋 루트 경로")
    parser.add_argument("--phase_index", type=int, default=30, 
                        help="Phase 컬럼의 시작 인덱스 (기본값: 30)")
    parser.add_argument("--shift_size", type=int, default=70,
                        help="Shifting window 크기 (기본값: 70)")
    parser.add_argument("--window_size", type=int, default=50,
                        help="Smoothing window 크기 (기본값: 50)")
    parser.add_argument("--method", choices=['sigmoid', 'linear'], default='sigmoid',
                        help="Smoothing 방법 (기본값: sigmoid)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="출력 디렉토리 (기본값: <root>_smoothed)")
    parser.add_argument("--inplace", action="store_true",
                        help="원본 파일에 직접 수정")
    parser.add_argument("--backup", action="store_true",
                        help="--inplace 사용 시 .bak 백업 생성")
    parser.add_argument("--dry_run", action="store_true",
                        help="실제로 저장하지 않고 통계만 출력")
    parser.add_argument("--pattern", default="*.parquet",
                        help="처리할 파일 패턴 (기본값: *.parquet)")
    
    args = parser.parse_args()
    
    # 경로 설정
    root = Path(args.root).resolve()
    if not root.exists():
        print(f"❌ 경로를 찾을 수 없습니다: {root}")
        return
    
    if args.inplace:
        out_root = root
    elif args.output_dir:
        out_root = Path(args.output_dir).resolve()
    else:
        out_root = Path(str(root) + "_smoothed").resolve()
    
    # Parquet 파일 찾기
    parquet_files = sorted(root.rglob(args.pattern))
    if not parquet_files:
        print(f"⚠️  '{args.pattern}' 패턴과 일치하는 파일이 없습니다: {root}")
        return
    
    print("=" * 70)
    print(f"Phase Smoothing 시작")
    print("=" * 70)
    print(f"📁 입력: {root}")
    print(f"📁 출력: {out_root}")
    print(f"🔧 설정:")
    print(f"   - Phase index: {args.phase_index}")
    print(f"   - Window size: {args.window_size}")
    print(f"   - Method: {args.method}")
    print(f"   - Files: {len(parquet_files)}")
    if args.dry_run:
        print(f"   - 💡 DRY RUN 모드 (저장 안 함)")
    print()
    
    # 각 파일 처리
    all_stats = []
    for src_path in parquet_files:
        rel_path = src_path.relative_to(root)
        dst_path = out_root / rel_path if not args.inplace else src_path
        
        # 백업 생성
        if args.inplace and args.backup and not args.dry_run:
            backup_path = src_path.with_suffix(src_path.suffix + '.bak')
            if not backup_path.exists():
                import shutil
                shutil.copy2(src_path, backup_path)
        
        stats = process_parquet_file(
            src_path, dst_path, args.phase_index, args.shift_size,
            args.window_size, args.method, args.dry_run
        )
        
        if stats:
            all_stats.append(stats)
            if stats['smoothed']:
                print(f"✅ {stats['file']}")
                print(f"   변화점: {stats['transitions']}개")
                print(f"   원본: {stats['original_unique']} unique values {stats['original_range']}")
                print(f"   결과: {stats['smoothed_unique']} unique values {stats['smoothed_range']}")
    
    # 최종 요약
    print()
    print("=" * 70)
    print("✅ 완료!")
    print("=" * 70)
    
    total_files = len(all_stats)
    smoothed_files = sum(1 for s in all_stats if s['smoothed'])
    total_transitions = sum(s['transitions'] for s in all_stats)
    
    print(f"📊 요약:")
    print(f"   - 전체 파일: {total_files}")
    print(f"   - Smoothing 적용: {smoothed_files}")
    print(f"   - 건너뜀: {total_files - smoothed_files}")
    print(f"   - 총 변화점: {total_transitions}")
    print()


if __name__ == "__main__":
    main()
