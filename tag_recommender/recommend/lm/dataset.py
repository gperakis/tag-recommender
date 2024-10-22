import logging

from sentence_transformers import InputExample
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TripletDataset(Dataset):
    """
    Dataset for triplet loss training.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.line_offsets = self._get_line_offsets()

    def _get_line_offsets(self) -> list:
        """
        Get byte offsets for each line in the file.

        Returns
        -------
        list
            A list of byte offsets for each line in the file.
        """
        line_offsets = []
        offset = 0
        with open(self.file_path, encoding="utf-8") as f:
            for line in f:
                line_offsets.append(offset)
                # Use 'utf-8' for accurate byte length
                offset += len(line.encode("utf-8"))
        return line_offsets

    def __len__(self) -> int:
        return len(self.line_offsets)

    def __getitem__(self, idx: int) -> InputExample | None:
        """
        Get an item from the dataset.

        Parameters
        ----------
        idx : int
            Index of the item to get.

        Returns
        -------
        InputExample | None
            An InputExample object if the item was successfully read,
            or None if the item was malformed.
        """
        with open(self.file_path, encoding="utf-8") as f:
            f.seek(self.line_offsets[idx])
            line = f.readline().strip()
            parts = line.split("\t")
            if len(parts) != 3:
                logger.warning(f"Malformed line at index {idx}: {line}")
                return None
            anchor, positive, negative = parts
            return InputExample(texts=[anchor, positive, negative])
