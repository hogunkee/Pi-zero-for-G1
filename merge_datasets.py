"""
LeRobot 데이터셋 병합 스크립트
하나의 폴더 안에 있는 여러 데이터셋을 하나로 합칩니다.
"""

import shutil
import json
import re
import os
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import pandas as pd
from tqdm import tqdm


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


def make_uniform(n: int, fps: float, unit: str, base=0.0, as_int=False):
    """FPS에 맞는 균일한 타임스탬프 생성"""
    t = (np.arange(n, dtype=np.float64) / float(fps)) * unit_scale(unit)
    t += float(base)
    if as_int:
        t = np.rint(t).astype(np.int64)
    return t


def filter_observation_state(state_value):
    """observation.state에서 앞 15개 dof 제거 (index 15부터만 유지)"""
    if isinstance(state_value, (list, np.ndarray)):
        return state_value[15:]
    return state_value


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

    if not parquet_dir.exists():
        return -1

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


def load_task_dict(tasks_decompose_path):
    """tasks_decompose.jsonl 로드하여 task_dict 생성"""
    task_dict = {}
    tasks_decompose_path = Path(tasks_decompose_path)

    if not tasks_decompose_path.exists():
        print(f"  ⚠ tasks_decompose.jsonl 파일을 찾을 수 없습니다: {tasks_decompose_path}")
        return task_dict

    with open(tasks_decompose_path, 'r', encoding='utf-8') as f:
        for line in f:
            x = json.loads(line)
            task_dict[x["task"]] = x["task_index"]

    return task_dict


def get_task_index_from_folder_name(folder_name, task_dict):
    """폴더 이름으로부터 task_index 찾기"""
    # 언더스코어를 공백으로 변환하여 매칭
    task_name = folder_name.replace("_", " ")

    if task_name in task_dict:
        return task_dict[task_name]

    # 매칭 실패시 None 반환
    return None


def get_state_dimension(parquet_file):
    """parquet 파일에서 observation.state의 dimension 확인"""
    table = pq.read_table(parquet_file)
    df = table.to_pandas()

    if "observation.state" in df.columns:
        first_state = df["observation.state"].iloc[0]
        if isinstance(first_state, (list, np.ndarray)):
            return len(first_state)

    return None


def merge_datasets(datasets_dir, output_dir, tasks_decompose_path="tasks_decompose.jsonl",
                   fps=None, timestamp_unit="s", timestamp_as_int=False):
    """
    하나의 폴더 내 여러 LeRobot 데이터셋을 병합

    Args:
        datasets_dir: 여러 데이터셋이 들어있는 폴더 경로
        output_dir: 출력 데이터셋 경로
        tasks_decompose_path: tasks_decompose.jsonl 파일 경로
        fps: FPS (타임스탬프 리타이밍용, None이면 리타이밍 안 함)
        timestamp_unit: 타임스탬프 단위 ("s", "ms", "us", "ns")
        timestamp_as_int: 타임스탬프를 int64로 저장할지 여부
    """
    datasets_dir = Path(datasets_dir)
    output_dir = Path(output_dir)

    print("=" * 70)
    print("LeRobot 데이터셋 병합 시작")
    print("=" * 70)

    # tasks_decompose.jsonl 로드
    print(f"\n[1] tasks_decompose.jsonl 로드 중...")
    task_dict = load_task_dict(tasks_decompose_path)
    print(f"  ✓ {len(task_dict)}개 태스크 정의 로드")

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # 데이터셋 폴더들 찾기
    print(f"\n[2] 데이터셋 검색 중...")
    dataset_folders = []
    for item in sorted(os.listdir(datasets_dir)):
        item_path = datasets_dir / item
        if item_path.is_dir():
            # meta 폴더가 있는지 확인
            if (item_path / "meta" / "info.json").exists():
                dataset_folders.append(item)

    print(f"  ✓ {len(dataset_folders)}개 데이터셋 발견: {dataset_folders}")

    if len(dataset_folders) == 0:
        raise ValueError(f"데이터셋을 찾을 수 없습니다: {datasets_dir}")

    # 병합 준비
    total_episodes = 0
    total_frames = 0
    total_videos = 0
    merged_episodes = []
    merged_tasks = None
    merged_modality = None
    merged_stats = None

    current_last_index = -1
    parquet_output_count = 0
    video_output_count = 0

    # 각 데이터셋 처리
    print(f"\n[3] 데이터셋 병합 중...")
    for dataset_idx, dataset_name in enumerate(dataset_folders):
        dataset_path = datasets_dir / dataset_name
        print(f"\n  [{dataset_idx + 1}/{len(dataset_folders)}] '{dataset_name}' 처리 중...")

        # task_index 찾기
        task_index = get_task_index_from_folder_name(dataset_name, task_dict)
        if task_index is None:
            print(f"    ⚠ task_index를 찾을 수 없어 스킵합니다: {dataset_name}")
            continue
        print(f"    ✓ task_index: {task_index}")

        # 메타데이터 로드
        meta = load_metadata(dataset_path / "meta")
        num_episodes = meta['info']['total_episodes']
        num_frames = meta['info']['total_frames']

        # Parquet 파일 처리
        src_parquet_dir = dataset_path / "data" / "chunk-000"
        dst_parquet_dir = output_dir / "data" / "chunk-000"

        if not src_parquet_dir.exists():
            print(f"    ⚠ parquet 파일이 없어 스킵합니다")
            continue

        parquet_files = sorted(src_parquet_dir.glob("*.parquet"))
        print(f"    📦 {len(parquet_files)}개 parquet 파일 처리 중...")

        # state dimension 확인 (첫 번째 파일로)
        state_dim = None
        if parquet_files:
            state_dim = get_state_dimension(parquet_files[0])
            if state_dim is not None:
                print(f"    📊 observation.state dimension: {state_dim}")
                if state_dim == 43:
                    print(f"    ✂️ state 필터링 적용 (앞 15개 제거)")

        for pf in tqdm(parquet_files, desc=f"    Processing", leave=False):
            # Parquet 파일 읽기
            table = pq.read_table(pf)
            df = table.to_pandas()

            # episode_index와 index 조정
            if 'episode_index' in df.columns:
                df['episode_index'] = df['episode_index'] + total_episodes
            if 'index' in df.columns:
                df['index'] = df['index'] + current_last_index + 1

            # task_index 설정
            if 'task_index' in df.columns:
                df['task_index'] = task_index

            # state 필터링 (dimension이 43일 때만)
            if state_dim == 43 and "observation.state" in df.columns:
                df["observation.state"] = df["observation.state"].apply(filter_observation_state)

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
                new_ep_num = ep_num + total_episodes
                old_pattern = match.group(0)
                new_pattern = f"episode_{new_ep_num:0{len(match.group(1))}d}"
                new_name = new_name.replace(old_pattern, new_pattern)

            # 저장
            table = pa.Table.from_pandas(df)
            pq.write_table(table, dst_parquet_dir / new_name)
            parquet_output_count += 1

        print(f"    ✓ {len(parquet_files)}개 parquet 파일 처리 완료")

        # 현재 데이터셋의 마지막 index 업데이트
        dataset_last_index = get_last_index_from_parquet(dataset_path)
        if dataset_last_index >= 0:
            current_last_index = current_last_index + 1 + dataset_last_index

        # Episodes 병합
        if 'episodes' in meta:
            for ep in meta['episodes']:
                ep_adjusted = ep.copy()
                ep_adjusted['episode_index'] = ep['episode_index'] + total_episodes
                # task_index 추가
                if 'task_index' not in ep_adjusted:
                    ep_adjusted['task_index'] = task_index
                merged_episodes.append(ep_adjusted)

        # Tasks 병합 (첫 번째 것만 사용)
        if merged_tasks is None and 'tasks' in meta:
            merged_tasks = meta['tasks']

        # Modality 병합 (첫 번째 것만 사용, state dimension 43이면 수정)
        if merged_modality is None and 'modality' in meta:
            merged_modality = meta['modality'].copy()
            # state dimension이 43이면 modality 수정
            if state_dim == 43 and 'state' in merged_modality:
                new_state = {
                    "left_arm": {"start": 0, "end": 7},
                    "right_arm": {"start": 7, "end": 14},
                    "left_hand": {"start": 14, "end": 21},
                    "right_hand": {"start": 21, "end": 28}
                }
                merged_modality["state"] = new_state

        # Stats 병합 (첫 번째 것만 사용)
        if merged_stats is None and 'stats' in meta:
            merged_stats = meta['stats']

        # 비디오 파일 복사
        video_src = dataset_path / "videos" / "chunk-000"
        video_dst = output_dir / "videos" / "chunk-000"

        video_count = 0
        if video_src.exists():
            for video_file in video_src.glob("*.mp4"):
                new_name = video_file.name

                # episode_XXXX 형태의 번호를 정규표현식으로 찾아서 조정
                match = re.search(r'episode_(\d+)', new_name)
                if match:
                    ep_num = int(match.group(1))
                    new_ep_num = ep_num + total_episodes
                    old_pattern = match.group(0)
                    new_pattern = f"episode_{new_ep_num:0{len(match.group(1))}d}"
                    new_name = new_name.replace(old_pattern, new_pattern)

                shutil.copy2(video_file, video_dst / new_name)
                video_count += 1
                video_output_count += 1

        if video_count > 0:
            print(f"    📹 {video_count}개 비디오 파일 복사 완료")

        # 누적
        total_episodes += num_episodes
        total_frames += num_frames
        total_videos += video_count

    # 병합된 메타데이터 생성
    print(f"\n[4] 메타데이터 생성 중...")

    # info.json 생성 (첫 번째 데이터셋 기반)
    first_dataset = datasets_dir / dataset_folders[0]
    first_meta = load_metadata(first_dataset / "meta")

    merged_info = first_meta['info'].copy()
    merged_info['total_episodes'] = total_episodes
    merged_info['total_frames'] = total_frames
    merged_info['total_videos'] = total_videos
    merged_info['total_chunks'] = 1
    merged_info['total_tasks'] = len(task_dict)

    # splits 정보 업데이트
    if 'splits' in merged_info:
        for split_name, split_value in merged_info['splits'].items():
            if isinstance(split_value, str) and ':' in split_value:
                if split_name == 'train':
                    merged_info['splits'][split_name] = f"0:{total_episodes}"
            elif isinstance(split_value, (int, float)):
                if split_name == 'train':
                    merged_info['splits'][split_name] = total_episodes

    # 메타데이터 저장
    print(f"\n[5] 메타데이터 저장 중...")

    with open(output_dir / "meta" / "info.json", 'w') as f:
        json.dump(merged_info, f, indent=2)
    print(f"  ✓ info.json 저장")

    if merged_episodes:
        with open(output_dir / "meta" / "episodes.jsonl", 'w') as f:
            for ep in merged_episodes:
                f.write(json.dumps(ep) + '\n')
        print(f"  ✓ episodes.jsonl 저장")

    if merged_tasks:
        with open(output_dir / "meta" / "tasks.jsonl", 'w') as f:
            for task in merged_tasks:
                f.write(json.dumps(task) + '\n')
        print(f"  ✓ tasks.jsonl 저장")

    if merged_stats:
        with open(output_dir / "meta" / "stats.json", 'w') as f:
            json.dump(merged_stats, f, indent=2)
        print(f"  ✓ stats.json 저장")

    if merged_modality:
        with open(output_dir / "meta" / "modality.json", 'w') as f:
            json.dump(merged_modality, f, indent=2)
        print(f"  ✓ modality.json 저장")

    print("\n" + "=" * 70)
    print("✅ 병합 완료!")
    print("=" * 70)
    print(f"\n📊 병합 결과:")
    print(f"  - 병합된 데이터셋: {len(dataset_folders)}개")
    print(f"  - 총 에피소드: {total_episodes}")
    print(f"  - 총 프레임: {total_frames}")
    print(f"  - Parquet 파일: {parquet_output_count}개")
    print(f"  - 비디오 파일: {video_output_count}개")
    print(f"  - 출력 경로: {output_dir}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LeRobot 데이터셋 병합 (여러 데이터셋)")
    parser.add_argument("--datasets_dir", type=str, required=True,
                        help="여러 데이터셋이 들어있는 폴더 경로")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="출력 데이터셋 경로")
    parser.add_argument("--tasks_decompose", type=str, default="tasks_decompose.jsonl",
                        help="tasks_decompose.jsonl 파일 경로 (기본값: tasks_decompose.jsonl)")

    # 타임스탬프 리타이밍 옵션
    parser.add_argument("--fps", type=float, default=50,
                        help="타임스탬프 리타이밍을 위한 FPS (지정하지 않으면 리타이밍 안 함)")
    parser.add_argument("--timestamp_unit", choices=["s", "ms", "us", "ns"], default="s",
                        help="타임스탬프 단위 (기본값: s)")
    parser.add_argument("--timestamp_as_int", action="store_true",
                        help="타임스탬프를 int64로 저장 (기본값: float64)")

    args = parser.parse_args()

    merge_datasets(
        args.datasets_dir,
        args.output_dir,
        tasks_decompose_path=args.tasks_decompose,
        fps=args.fps,
        timestamp_unit=args.timestamp_unit,
        timestamp_as_int=args.timestamp_as_int
    )

# python merge_datasets.py \
#   --datasets_dir /data1/hogun/dataset/1205_TaskDecompose \
#   --output_dir /data1/hogun/dataset/merged_output \
#   --tasks_decompose tasks_decompose.jsonl \
#   --fps 30