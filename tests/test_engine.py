import tempfile
import unittest
from pathlib import Path

from geni.engine import LocalStore


class LocalStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = LocalStore(Path(self.temporary_directory.name) / "geni.sqlite3")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_persists_conversations_and_searchable_messages(self):
        conversation_id = self.store.conversations()[0]["id"]
        self.store.rename_conversation(conversation_id, "Svenska")
        self.store.add_message(conversation_id, "user", "Vad är vatten?")
        self.assertEqual(self.store.conversations("vatten")[0]["id"], conversation_id)
        reopened = LocalStore(self.store.path)
        self.assertEqual(reopened.messages(conversation_id)[0]["content"], "Vad är vatten?")

    def test_only_approved_examples_are_in_training_material(self):
        self.store.add_document("exempel.txt", "Svensk text för träning.")
        example_id = self.store.add_example("Fråga", "Svar")
        self.assertEqual(list(self.store.training_texts()), ["Svensk text för träning."])
        self.store.approve_example(example_id)
        self.assertEqual(len(list(self.store.training_texts())), 2)

    def test_rejects_empty_examples_and_invalid_roles(self):
        with self.assertRaises(ValueError):
            self.store.add_example(" ", "svar")
        with self.assertRaises(ValueError):
            self.store.add_message(self.store.conversations()[0]["id"], "system", "text")


if __name__ == "__main__":
    unittest.main()
