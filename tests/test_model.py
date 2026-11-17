import tempfile
import threading
import unittest
from pathlib import Path

import torch

from geni.model import ByteTransformer, generate_stream, load_checkpoint, train_model


class ModelTests(unittest.TestCase):
    def test_model_trains_generates_and_loads_checkpoint(self):
        torch.manual_seed(1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            model, losses = train_model(
                ["Svensk text att lära sig från. " * 8],
                checkpoint,
                steps=2,
                batch_size=2,
                context=16,
            )
            self.assertEqual(len(losses), 2)
            self.assertTrue(checkpoint.exists())
            loaded = load_checkpoint(checkpoint)
            output = list(generate_stream(loaded, "Hej", max_new_tokens=3))
            self.assertEqual(len(output), 3)
            self.assertIsInstance(model, ByteTransformer)

    def test_generation_observes_cancellation(self):
        model = ByteTransformer(context=16)
        cancel = threading.Event()
        cancel.set()
        self.assertEqual(list(generate_stream(model, "Hej", cancel=cancel)), [])


if __name__ == "__main__":
    unittest.main()
