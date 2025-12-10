"""
LeRobot 데이터셋 재인덱싱 스크립트
변화점이 있는 데이터만 남긴 후 인덱스를 연속적으로 재정렬합니다.
"""

import argparse
import json
import re
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


def extract_episode_number(filename):
    """파일명에서 에피소드 번호 추출"""
    match = re.search(r'episode_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def reindex_dataset(input_dir, output_dir, dry_run=False):
    """
    데이터셋을 재인덱싱합니다.

    Args:
        input_dir: 입력 데이터셋 경로 (smooth_phase.py 출력)
        output_dir: 재인덱싱된 데이터셋을 저장할 경로
        dry_run: True면 저장하지 않고 통계만 출력
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    print("=" * 70)
    print("LeRobot 데이터셋 재인덱싱 시작")
    print("=" * 70)
    print(f"📁 입력: {input_dir}")
    print(f"📁 출력: {output_dir}")
    if dry_run:
        print("💡 DRY RUN 모드 (저장 안 함)")
    print()

    # 디렉토리 확인
    parquet_dir = input_dir / "data" / "chunk-000"
    video_dir = input_dir / "videos" / "chunk-000"
    meta_dir = input_dir / "meta"

    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet 디렉토리를 찾을 수 없습니다: {parquet_dir}")

    # 출력 디렉토리 생성
    if not dry_run:
        (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (output_dir / "videos" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (output_dir / "meta").mkdir(parents=True, exist_ok=True)

    # 1. Parquet 파일 목록 가져오기 및 정렬
    print("[1/4] Parquet 파일 스캔 중...")
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    print(f"  ✓ {len(parquet_files)}개 파일 발견")

    # 2. 에피소드 인덱스 매핑 생성
    print("\n[2/4] 에피소드 재인덱싱 중...")
    old_to_new_episode = {}
    new_episode_idx = 0

    for pf in parquet_files:
        old_ep_num = extract_episode_number(pf.name)
        if old_ep_num is not None and old_ep_num not in old_to_new_episode:
            old_to_new_episode[old_ep_num] = new_episode_idx
            new_episode_idx += 1

    print(f"  ✓ 기존 에피소드: {len(old_to_new_episode)}개")
    print(f"  ✓ 새 에피소드: {new_episode_idx}개")

    # 3. Parquet 파일 재인덱싱 및 저장
    print("\n[3/4] Parquet 파일 재인덱싱 중...")

    total_frames = 0
    new_index = 0
    episode_info = {}  # {new_episode_idx: {'length': N, 'old_idx': M}}

    for pf in sorted(parquet_files):
        old_ep_num = extract_episode_number(pf.name)
        if old_ep_num is None:
            print(f"  ⚠️  파일명에서 에피소드 번호를 찾을 수 없음: {pf.name}")
            continue

        new_ep_num = old_to_new_episode[old_ep_num]

        # Parquet 파일 읽기
        table = pq.read_table(pf)
        df = table.to_pandas()

        # 프레임 수 카운트
        num_frames = len(df)
        total_frames += num_frames

        # 에피소드 정보 저장
        if new_ep_num not in episode_info:
            episode_info[new_ep_num] = {
                'length': num_frames,
                'old_idx': old_ep_num,
                'start_index': new_index
            }

        # episode_index 재설정
        if 'episode_index' in df.columns:
            df['episode_index'] = new_ep_num

        # index 재설정 (연속적으로)
        if 'index' in df.columns:
            df['index'] = list(range(new_index, new_index + num_frames))

        new_index += num_frames

        # 새 파일명 생성
        new_filename = f"episode_{new_ep_num:06d}.parquet"

        # 저장
        if not dry_run:
            table = pa.Table.from_pandas(df)
            pq.write_table(table, output_dir / "data" / "chunk-000" / new_filename)

        print(f"  ✓ episode_{old_ep_num:06d}.parquet -> {new_filename} ({num_frames} frames)")

    print(f"\n  📊 총 {total_frames}개 프레임")

    # 4. 비디오 파일 복사 (에피소드 번호 매칭)
    print("\n[4/4] 비디오 파일 복사 중...")

    if video_dir.exists():
        video_files = list(video_dir.glob("*.mp4"))
        copied_videos = 0

        for vf in video_files:
            old_ep_num = extract_episode_number(vf.name)
            if old_ep_num is None:
                print(f"  ⚠️  파일명에서 에피소드 번호를 찾을 수 없음: {vf.name}")
                continue

            # 매핑에 없는 에피소드는 건너뜀 (smooth_phase에서 제거된 것)
            if old_ep_num not in old_to_new_episode:
                print(f"  ⏭️  건너뜀 (parquet 없음): {vf.name}")
                continue

            new_ep_num = old_to_new_episode[old_ep_num]

            # 새 파일명 생성
            new_name = re.sub(
                r'episode_(\d+)',
                f'episode_{new_ep_num:06d}',
                vf.name
            )

            if not dry_run:
                shutil.copy2(vf, output_dir / "videos" / "chunk-000" / new_name)

            copied_videos += 1
            print(f"  ✓ {vf.name} -> {new_name}")

        print(f"\n  📊 {copied_videos}개 비디오 복사 완료")
    else:
        print("  ⚠️  비디오 디렉토리를 찾을 수 없습니다")

    # 5. 메타데이터 생성
    print("\n[5/5] 메타데이터 생성 중...")

    if meta_dir.exists():
        # 기존 메타데이터 로드
        info_path = meta_dir / "info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
        else:
            info = {}

        # episodes.jsonl 로드 (있으면)
        episodes_path = meta_dir / "episodes.jsonl"
        old_episodes = []
        if episodes_path.exists():
            with open(episodes_path, 'r') as f:
                for line in f:
                    old_episodes.append(json.loads(line))

        # tasks.jsonl 로드 (있으면)
        tasks_path = meta_dir / "tasks.jsonl"
        tasks = []
        if tasks_path.exists():
            with open(tasks_path, 'r') as f:
                for line in f:
                    tasks.append(json.loads(line))

        # info.json 업데이트
        info['total_episodes'] = new_episode_idx
        info['total_videos'] = new_episode_idx
        info['total_frames'] = total_frames
        info['total_chunks'] = 1

        # splits 정보 업데이트
        if 'splits' in info:
            for split_name in info['splits']:
                if split_name == 'train':
                    if isinstance(info['splits'][split_name], str) and ':' in info['splits'][split_name]:
                        info['splits'][split_name] = f"0:{new_episode_idx}"
                    else:
                        info['splits'][split_name] = new_episode_idx

        # episodes.jsonl 재생성
        new_episodes = []
        for new_ep_idx in sorted(episode_info.keys()):
            old_ep_idx = episode_info[new_ep_idx]['old_idx']

            # 기존 에피소드 정보 찾기
            old_ep_info = None
            for ep in old_episodes:
                if ep.get('episode_index') == old_ep_idx:
                    old_ep_info = ep.copy()
                    break

            if old_ep_info:
                old_ep_info['episode_index'] = new_ep_idx
                new_episodes.append(old_ep_info)
            else:
                # 기존 정보가 없으면 기본 정보만 생성
                new_episodes.append({
                    'episode_index': new_ep_idx,
                    'length': episode_info[new_ep_idx]['length']
                })

        # 저장
        if not dry_run:
            # info.json 저장
            with open(output_dir / "meta" / "info.json", 'w') as f:
                json.dump(info, f, indent=2)
            print("  ✓ info.json 저장")

            # episodes.jsonl 저장
            with open(output_dir / "meta" / "episodes.jsonl", 'w') as f:
                for ep in new_episodes:
                    f.write(json.dumps(ep) + '\n')
            print("  ✓ episodes.jsonl 저장")

            # tasks.jsonl 복사 (변경 없음)
            if tasks:
                with open(output_dir / "meta" / "tasks.jsonl", 'w') as f:
                    for task in tasks:
                        f.write(json.dumps(task) + '\n')
                print("  ✓ tasks.jsonl 저장")

            # 기타 메타데이터 파일 복사 (stats.json, modality.json 등)
            for meta_file in meta_dir.glob("*.json"):
                if meta_file.name not in ['info.json']:
                    shutil.copy2(meta_file, output_dir / "meta" / meta_file.name)
                    print(f"  ✓ {meta_file.name} 복사")
    else:
        print("  ⚠️  메타데이터 디렉토리를 찾을 수 없습니다")

    # 최종 요약
    print("\n" + "=" * 70)
    print("✅ 재인덱싱 완료!")
    print("=" * 70)
    print(f"\n📊 요약:")
    print(f"  - 총 에피소드: {new_episode_idx}")
    print(f"  - 총 프레임: {total_frames}")
    print(f"  - Parquet 파일: {len(parquet_files)}개")
    print(f"  - 출력 경로: {output_dir}")

    if old_to_new_episode:
        removed_count = max(old_to_new_episode.keys()) + 1 - len(old_to_new_episode)
        if removed_count > 0:
            print(f"  - 제거된 에피소드: {removed_count}개 (변화점 없음)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="LeRobot 데이터셋 재인덱싱 (smooth_phase 이후)"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="입력 데이터셋 경로 (smooth_phase 출력)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="출력 디렉토리 (기본값: <input_dir>_reindexed)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="실제로 저장하지 않고 통계만 출력"
    )

    args = parser.parse_args()

    # 출력 디렉토리 설정
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(Path(args.input_dir).resolve()) + "_reindexed"

    reindex_dataset(args.input_dir, output_dir, args.dry_run)


if __name__ == "__main__":
    main()
