import pyarrow.parquet as pq
import pyarrow as pa
import os
import math

def split_parquet(input_path: str, output_dir: str, n: int):
    os.makedirs(output_dir, exist_ok=True)

    # 读取 parquet 文件
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows

    rows_per_file = math.ceil(total_rows / n)

    print(f"Total rows: {total_rows}")
    print(f"Rows per split: {rows_per_file}")

    current_chunk = []
    current_count = 0
    file_idx = 0

    for batch in parquet_file.iter_batches():
        table = pa.Table.from_batches([batch])
        current_chunk.append(table)
        current_count += table.num_rows

        # 满足切分条件就写入文件
        while current_count >= rows_per_file:
            combined = pa.concat_tables(current_chunk)
            to_write = combined.slice(0, rows_per_file)

            output_path = os.path.join(output_dir, f"part_{file_idx}.parquet")
            pq.write_table(to_write, output_path)
            print(f"Write: {output_path}")

            file_idx += 1

            # 剩余部分继续
            remaining = combined.slice(rows_per_file)
            current_chunk = [remaining] if remaining.num_rows > 0 else []
            current_count = remaining.num_rows

    # 写最后一块
    if current_count > 0:
        combined = pa.concat_tables(current_chunk)
        output_path = os.path.join(output_dir, f"part_{file_idx}.parquet")
        pq.write_table(combined, output_path)
        print(f"Write: {output_path}")


if __name__ == "__main__":
    INPUT_PARQUET = "/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_data/sft_seed/v3/sft-hotpot10000-nq10000-seed.parquet"
    OUTPUT_DIR = "/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_data/sft_seed/v3/divide"

    split_parquet(
        input_path=INPUT_PARQUET,
        output_dir=OUTPUT_DIR,
        n=5
    )