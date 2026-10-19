"""Benchmark images must not be silently replaced by neighbouring samples."""
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from eval_suite.datasets.biota import BiotaDataset


class BiotaIntegrityTests(unittest.TestCase):
    def test_missing_record_raises_instead_of_returning_neighbours_image_and_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            Image.new('RGB',(8,8)).save(root/'present.jpg')
            dataset=BiotaDataset([(str(root/'missing.jpg'),[0]),(str(root/'present.jpg'),[1])],2)
            with self.assertRaisesRegex(FileNotFoundError,'missing.jpg'):
                dataset[0]
            image,label=dataset[1]
            self.assertEqual(label.tolist(),[0.,1.])


if __name__=='__main__':unittest.main()
