import unittest

import grain

from dreamer.tokenization import make_tokenization_batch


class TokenizationBatchingTest(unittest.TestCase):
    def test_preserves_final_partial_batch(self):
        source = grain.MapDataset.source(range(10))

        batches = list(source.apply(make_tokenization_batch(batch_size=8)))

        self.assertEqual([len(batch) for batch in batches], [8, 2])
        self.assertEqual(
            [int(record) for batch in batches for record in batch],
            list(range(10)),
        )


if __name__ == "__main__":
    unittest.main()
