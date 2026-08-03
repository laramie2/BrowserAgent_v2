from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from RL.prepare_seed_data import main


def seed_frame(source: str, count: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "data_source": source,
                "extra_info": {
                    "index": index,
                    "question": f"{source} question {index}",
                    "golden_answers": [f"answer {index}"],
                },
                "reward_model": {"style": "rule"},
                "prompt": [],
            }
            for index in range(count)
        ]
    )


class PrepareSeedDataTest(unittest.TestCase):
    def test_builds_train_and_balanced_validation_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            for dataset, validation_name in (
                ("nq", "test-00000-of-00001.parquet"),
                ("hotpot", "validation-00000-of-00001.parquet"),
            ):
                dataset_dir = source / dataset
                dataset_dir.mkdir(parents=True)
                frame = seed_frame(dataset)
                frame.to_parquet(dataset_dir / "train-00000-of-00001.parquet")
                frame.to_parquet(dataset_dir / validation_name)

            status = main(
                [
                    "--source-root",
                    str(source),
                    "--output-root",
                    str(output),
                    "--num-samples",
                    "2",
                    "--validation-samples",
                    "2",
                    "--no-exclude",
                ]
            )

            self.assertEqual(status, 0)
            self.assertEqual(len(pd.read_parquet(output / "nq/train_2_labelled.parquet")), 2)
            self.assertEqual(
                len(pd.read_parquet(output / "hotpot/train_2_labelled.parquet")), 2
            )
            validation = pd.read_parquet(output / "validation_2/data.parquet")
            self.assertEqual(len(validation), 2)
            self.assertEqual(set(validation["data_source"]), {"nq", "hotpot"})


if __name__ == "__main__":
    unittest.main()
