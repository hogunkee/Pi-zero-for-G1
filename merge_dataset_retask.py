"""
LeRobot 데이터셋 병합 스크립트
두 개의 LeRobot 데이터셋을 하나로 합칩니다.
"""

import shutil
import json
import re
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import pandas as pd


def unit_scale(unit: str) -> float:
    """시간 단위를 스케일로 변환"""
    return {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}[unit]


def detect_timestamp_cols(df: pd.DataFrame) -> list:
    """타임스탬프 컬럼 자동 감지"""
    cols = []
    for c in df.columns:
        cl = str(c).lower()
        if "timestamp" in cl or cl.endswith("_ts"):
            cols.append(c)
    return cols


def detect_task_cols(df: pd.DataFrame) -> list:
    """태스크 컬럼 자동 감지"""
    cols = []
    for c in df.columns:
        cl = str(c).lower()
        if "task" in cl or cl.endswith("_ts"):
            cols.append(c)
    return cols


def make_uniform(n: int, fps: float, unit: str, base=0.0, as_int=False):
    """FPS에 맞는 균일한 타임스탬프 생성"""
    t = (np.arange(n, dtype=np.float64) / float(fps)) * unit_scale(unit)
    t += float(base)
    if as_int:
        t = np.rint(t).astype(np.int64)
    return t


def load_metadata(meta_dir):
    """메타데이터 로드"""
    meta_dir = Path(meta_dir)
    metadata = {}
    
    # info.json 로드
    info_path = meta_dir / "info.json"
    if info_path.exists():
        with open(info_path, 'r') as f:
            metadata['info'] = json.load(f)
    
    # episodes.jsonl 로드
    episodes_path = meta_dir / "episodes.jsonl"
    if episodes_path.exists():
        episodes = []
        with open(episodes_path, 'r') as f:
            for line in f:
                episodes.append(json.loads(line))
        metadata['episodes'] = episodes
    
    # tasks.jsonl 로드 (있을 경우)
    tasks_path = meta_dir / "tasks.jsonl"
    if tasks_path.exists():
        tasks = []
        with open(tasks_path, 'r') as f:
            for line in f:
                tasks.append(json.loads(line))
        metadata['tasks'] = tasks
    
    # stats.json 로드 (있을 경우)
    stats_path = meta_dir / "stats.json"
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            metadata['stats'] = json.load(f)
    
    # modality.json 로드 (있을 경우)
    modality_path = meta_dir / "modality.json"
    if modality_path.exists():
        with open(modality_path, 'r') as f:
            metadata['modality'] = json.load(f)
    
    return metadata


def get_last_index_from_parquet(data_dir):
    """데이터셋의 마지막 index 값을 찾기"""
    data_dir = Path(data_dir)
    parquet_dir = data_dir / "data" / "chunk-000"
    
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        return -1
    
    # 마지막 parquet 파일 읽기
    last_file = parquet_files[-1]
    table = pq.read_table(last_file)
    
    # index 컬럼의 마지막 값 반환
    if 'index' in table.column_names:
        index_column = table.column('index').to_pylist()
        return max(index_column) if index_column else -1
    
    return -1


def merge_datasets(data_dir1, data_dir2, output_dir, fps=None, timestamp_unit="s", timestamp_as_int=False):
    """
    두 개의 LeRobot 데이터셋을 병합
    
    Args:
        data_dir1: 첫 번째 데이터셋 경로
        data_dir2: 두 번째 데이터셋 경로
        output_dir: 출력 데이터셋 경로
        fps: FPS (타임스탬프 리타이밍용, None이면 리타이밍 안 함)
        timestamp_unit: 타임스탬프 단위 ("s", "ms", "us", "ns")
        timestamp_as_int: 타임스탬프를 int64로 저장할지 여부
    """
    data_dir1 = Path(data_dir1)
    data_dir2 = Path(data_dir2)
    output_dir = Path(output_dir)
    
    print("=" * 60)
    print("LeRobot 데이터셋 병합 시작")
    print("=" * 60)
    
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos" / "chunk-000").mkdir(parents=True, exist_ok=True)
    
    # 1. 데이터셋 확인
    print("\n[1/5] 데이터셋 확인 중...")
    print(f"  - Dataset 1: {data_dir1}")
    parquet_files1 = list((data_dir1 / "data" / "chunk-000").glob("*.parquet"))
    if not parquet_files1:
        raise FileNotFoundError(f"No parquet files found in {data_dir1 / 'data' / 'chunk-000'}")
    print(f"    ✓ {len(parquet_files1)}개 parquet 파일")
    
    print(f"  - Dataset 2: {data_dir2}")
    parquet_files2 = list((data_dir2 / "data" / "chunk-000").glob("*.parquet"))
    if not parquet_files2:
        raise FileNotFoundError(f"No parquet files found in {data_dir2 / 'data' / 'chunk-000'}")
    print(f"    ✓ {len(parquet_files2)}개 parquet 파일")
    
    # 2. 메타데이터 로드
    print("\n[2/5] 메타데이터 로드 중...")
    meta1 = load_metadata(data_dir1 / "meta")
    meta2 = load_metadata(data_dir2 / "meta")
    
    # 3. Parquet 파일 복사
    print("\n[3/5] Parquet 파일 복사 중...")
    
    num_episodes1 = meta1['info']['total_episodes']
    num_frames1 = meta1['info']['total_frames']
    
    # Dataset1의 마지막 index 찾기
    last_index1 = get_last_index_from_parquet(data_dir1)
    print(f"  📌 Dataset1의 마지막 index: {last_index1}")
    
    # Dataset1의 parquet 파일 복사 및 타임스탬프 조정
    src1_parquet = data_dir1 / "data" / "chunk-000"
    dst_parquet = output_dir / "data" / "chunk-000"
    
    parquet_files1 = sorted(src1_parquet.glob("*.parquet"))
    
    for pf in parquet_files1:
        # Parquet 파일 읽기
        table = pq.read_table(pf)
        df = table.to_pandas()

        # 태스크 0
        # task_cols = detect_task_cols(df)
        # for col in task_cols:
        #     df[col] = [0] * len(df)

        if fps is not None:
            # 타임스탬프 컬럼 찾기
            ts_cols = detect_timestamp_cols(df)
            if ts_cols:
                # 각 타임스탬프 컬럼 리타이밍
                for col in ts_cols:
                    df[col] = make_uniform(len(df), fps, timestamp_unit, base=0.0, as_int=timestamp_as_int)
            
        # 다시 parquet로 저장
        table = pa.Table.from_pandas(df)
        pq.write_table(table, dst_parquet / pf.name)
    print(f"  ✓ Dataset1: {len(parquet_files1)}개 파일 복사 완료 (타임스탬프 리타이밍)")
    
    # Dataset2의 parquet 파일 복사 및 내부 데이터 조정
    src2_parquet = data_dir2 / "data" / "chunk-000"
    parquet_files2 = sorted(src2_parquet.glob("*.parquet"))
    
    for pf in parquet_files2:
        # Parquet 파일 읽기
        table = pq.read_table(pf)
        df = table.to_pandas()
        
        # episode_index와 index 조정
        if 'episode_index' in df.columns:
            df['episode_index'] = df['episode_index'] + num_episodes1
        if 'index' in df.columns:
            df['index'] = df['index'] + last_index1 + 1

        # 태스크 1
        # task_cols = detect_task_cols(df)
        # for col in task_cols:
        #     df[col] = [1] * len(df)
        
        # 타임스탬프 리타이밍 (fps가 지정된 경우)
        if fps is not None:
            ts_cols = detect_timestamp_cols(df)
            if ts_cols:
                for col in ts_cols:
                    df[col] = make_uniform(len(df), fps, timestamp_unit, base=0.0, as_int=timestamp_as_int)
        
        # 새로운 파일명 생성
        new_name = pf.name
        match = re.search(r'episode_(\d+)', new_name)
        if match:
            ep_num = int(match.group(1))
            new_ep_num = ep_num + num_episodes1
            old_pattern = match.group(0)
            new_pattern = f"episode_{new_ep_num:0{len(match.group(1))}d}"
            new_name = new_name.replace(old_pattern, new_pattern)
        
        # 수정된 parquet 파일 저장
        table = pa.Table.from_pandas(df)
        pq.write_table(table, dst_parquet / new_name)
    
    if fps is not None:
        print(f"  ✓ Dataset2: {len(parquet_files2)}개 파일 복사 완료 (episode_index, index, 타임스탬프 조정)")
    else:
        print(f"  ✓ Dataset2: {len(parquet_files2)}개 파일 복사 완료 (episode_index 및 index 조정)")
    print(f"  ✓ 총 {len(parquet_files1) + len(parquet_files2)}개 파일")
    
    # 4. 메타데이터 병합
    print("\n[4/5] 메타데이터 병합 중...")
    merged_meta = {}
    
    # info.json 병합
    merged_meta['info'] = meta1['info'].copy()
    merged_meta['info']['total_episodes'] = num_episodes1 + meta2['info']['total_episodes']
    merged_meta['info']['total_frames'] = num_frames1 + meta2['info']['total_frames']
    merged_meta['info']['total_tasks'] = meta1['info'].get('total_tasks', 0) + meta2['info'].get('total_tasks', 0)
    merged_meta['info']['total_videos'] = meta1['info'].get('total_videos', 0) + meta2['info'].get('total_videos', 0)
    merged_meta['info']['total_chunks'] = 1  # 하나의 chunk로 병합
    
    # splits 정보 업데이트
    if 'splits' in merged_meta['info']:
        for split_name, split_value in merged_meta['info']['splits'].items():
            # splits가 "0:100" 형태의 문자열인 경우
            if isinstance(split_value, str) and ':' in split_value:
                parts = split_value.split(':')
                if len(parts) == 2:
                    start, end = int(parts[0]), int(parts[1])
                    # train split은 0부터 전체 에피소드까지
                    if split_name == 'train':
                        merged_meta['info']['splits'][split_name] = f"0:{merged_meta['info']['total_episodes']}"
            # splits가 숫자인 경우
            elif isinstance(split_value, (int, float)):
                if split_name == 'train':
                    merged_meta['info']['splits'][split_name] = merged_meta['info']['total_episodes']
    
    # episodes.jsonl 병합
    if 'episodes' in meta1 and 'episodes' in meta2:
        merged_episodes = meta1['episodes'].copy()
        for ep in meta2['episodes']:
            ep_adjusted = ep.copy()
            ep_adjusted['episode_index'] = ep['episode_index'] + num_episodes1
            merged_episodes.append(ep_adjusted)
        merged_meta['episodes'] = merged_episodes
    
    # tasks.jsonl 병합 (있을 경우)
    merged_meta['tasks'] = meta1['tasks']
    # if 'tasks' in meta1 and 'tasks' in meta2:
    #     merged_meta['tasks'] = meta1['tasks'] + meta2['tasks']
    # elif 'tasks' in meta1:
    #     merged_meta['tasks'] = meta1['tasks']
    # elif 'tasks' in meta2:
    #     merged_meta['tasks'] = meta2['tasks']
    
    # stats.json은 새로 계산해야 하므로 일단 dataset1의 것을 사용
    if 'stats' in meta1:
        merged_meta['stats'] = meta1['stats']
        print("  ⚠ stats.json은 dataset1의 값을 사용합니다. 필요시 재계산하세요.")
    
    # modality.json 병합
    if 'modality' in meta1 and 'modality' in meta2:
        # 두 데이터셋의 modality가 동일한지 확인
        if meta1['modality'] == meta2['modality']:
            merged_meta['modality'] = meta1['modality']
        else:
            print("  ⚠ 두 데이터셋의 modality가 다릅니다. dataset1의 modality를 사용합니다.")
            merged_meta['modality'] = meta1['modality']
    elif 'modality' in meta1:
        merged_meta['modality'] = meta1['modality']
    elif 'modality' in meta2:
        merged_meta['modality'] = meta2['modality']
    
    # 5. 메타데이터 저장
    print("\n[5/5] 메타데이터 저장 중...")
    
    # 메타데이터 저장
    with open(output_dir / "meta" / "info.json", 'w') as f:
        json.dump(merged_meta['info'], f, indent=2)
    print(f"  ✓ info.json 저장")
    
    if 'episodes' in merged_meta:
        with open(output_dir / "meta" / "episodes.jsonl", 'w') as f:
            for ep in merged_meta['episodes']:
                f.write(json.dumps(ep) + '\n')
        print(f"  ✓ episodes.jsonl 저장")
    
    if 'tasks' in merged_meta:
        with open(output_dir / "meta" / "tasks.jsonl", 'w') as f:
            for task in merged_meta['tasks']:
                f.write(json.dumps(task) + '\n')
        print(f"  ✓ tasks.jsonl 저장")
    
    if 'stats' in merged_meta:
        with open(output_dir / "meta" / "stats.json", 'w') as f:
            json.dump(merged_meta['stats'], f, indent=2)
        print(f"  ✓ stats.json 저장")
    
    if 'modality' in merged_meta:
        with open(output_dir / "meta" / "modality.json", 'w') as f:
            json.dump(merged_meta['modality'], f, indent=2)
        print(f"  ✓ modality.json 저장")
    
    # videos 복사
    print("\n[6/6] 비디오 파일 복사 중...")
    video_src1 = data_dir1 / "videos" / "chunk-000"
    video_src2 = data_dir2 / "videos" / "chunk-000"
    video_dst = output_dir / "videos" / "chunk-000"
    
    # dataset1의 비디오 복사
    video_count1 = 0
    if video_src1.exists():
        for video_file in video_src1.glob("*.mp4"):
            shutil.copy2(video_file, video_dst / video_file.name)
            video_count1 += 1
        print(f"  ✓ Dataset1: {video_count1}개 비디오 복사 완료")
    
    # dataset2의 비디오 복사 (이름 조정)
    video_count2 = 0
    if video_src2.exists():
        for video_file in video_src2.glob("*.mp4"):
            new_name = video_file.name
            
            # episode_XXXX 형태의 번호를 정규표현식으로 찾아서 조정
            # 예: episode_0093.mp4 -> episode_0193.mp4 (100 추가)
            match = re.search(r'episode_(\d+)', new_name)
            if match:
                ep_num = int(match.group(1))
                new_ep_num = ep_num + num_episodes1
                # 원래 자릿수를 유지하면서 번호 교체
                old_pattern = match.group(0)  # episode_0093
                new_pattern = f"episode_{new_ep_num:0{len(match.group(1))}d}"
                new_name = new_name.replace(old_pattern, new_pattern)
            
            shutil.copy2(video_file, video_dst / new_name)
            video_count2 += 1
        print(f"  ✓ Dataset2: {video_count2}개 비디오 복사 완료 (episode 번호 조정)")
    
    print(f"  ✓ 총 {video_count1 + video_count2}개 비디오")
    
    print("\n" + "=" * 60)
    print("✅ 병합 완료!")
    print("=" * 60)
    print(f"\n📊 병합 결과:")
    print(f"  - 총 에피소드: {merged_meta['info']['total_episodes']}")
    print(f"  - 총 프레임: {merged_meta['info']['total_frames']}")
    print(f"  - Parquet 파일: {len(parquet_files1) + len(parquet_files2)}개")
    print(f"  - 비디오 파일: {video_count1 + video_count2}개")
    print(f"  - 출력 경로: {output_dir}")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LeRobot 데이터셋 병합")
    parser.add_argument("--data_dir1", type=str, required=True, help="첫 번째 데이터셋 경로")
    parser.add_argument("--data_dir2", type=str, required=True, help="두 번째 데이터셋 경로")
    parser.add_argument("--output_dir", type=str, required=True, help="출력 데이터셋 경로")
    
    # 타임스탬프 리타이밍 옵션
    parser.add_argument("--fps", type=float, default=None, help="타임스탬프 리타이밍을 위한 FPS (지정하지 않으면 리타이밍 안 함)")
    parser.add_argument("--timestamp_unit", choices=["s", "ms", "us", "ns"], default="s", 
                        help="타임스탬프 단위 (기본값: s)")
    parser.add_argument("--timestamp_as_int", action="store_true", 
                        help="타임스탬프를 int64로 저장 (기본값: float64)")
    
    args = parser.parse_args()
    
    merge_datasets(
        args.data_dir1, 
        args.data_dir2, 
        args.output_dir,
        fps=args.fps,
        timestamp_unit=args.timestamp_unit,
        timestamp_as_int=args.timestamp_as_int
    )
