from __future__ import annotations

import sys
import unittest
from pathlib import Path

test_dir = Path(__file__).resolve().parent
source_dir = test_dir.parents[0] / "src"
if not source_dir.is_dir():
    source_dir = test_dir
sys.path.insert(0, str(source_dir))

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import DoubleType, StringType, StructField, StructType
except ModuleNotFoundError:
    SparkSession = None

if SparkSession is not None:
    from sagemaker_spark_job import candidate_output_frame
else:
    candidate_output_frame = None


@unittest.skipIf(SparkSession is None, "PySpark integration check runs on the cloud worker")
class SparkCandidateOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("candidate-output-contract-test")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_output_keeps_all_retrieved_candidates_before_model_filtering(self):
        query_ids = self.spark.createDataFrame([("q1",), ("q2",)], ["source1_id"])
        schema = StructType([
            StructField("source1_id", StringType(), False),
            StructField("target_source", StringType(), False),
            StructField("target_id", StringType(), False),
            StructField("retrieval_score", DoubleType(), False),
        ])
        generated = self.spark.createDataFrame([
            ("q1", "S2", "m2", 8.0),
            ("q1", "S2", "m1", 10.0),
            ("q1", "S3", "m3", 9.0),
        ], schema)

        rows = {
            row["source1_entity_id"]: row["candidate_entity_ids"]
            for row in candidate_output_frame(query_ids, generated).collect()
        }
        self.assertEqual(rows, {"q1": "m1,m3,m2", "q2": ""})


if __name__ == "__main__":
    unittest.main()
