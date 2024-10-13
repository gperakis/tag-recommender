import mmh3
import numpy as np


class CountMinSketch:
    """Simple Count-Min Sketch implementation for frequency estimation"""

    def __init__(self, width: int = 10_000, depth: int = 5):
        """
        Initialize the Count-Min Sketch with the given width and depth.

        Parameters
        ----------
        width : int (default: 10000)
            Number of columns in the sketch
        depth : int (default: 5)
            Number of hash functions (rows). More rows reduce the error rate.
        """
        self.width = width
        self.depth = depth  # Number of hash functions (rows)
        self.table = np.zeros((depth, width))

        # Different seed for each hash function
        self.hash_seeds = [i for i in range(depth)]

    def update(self, key: bytes, count: int = 1):
        """
        Update the Count-Min Sketch with the given key and count.

        Parameters
        ----------
        key : bytes
        count : int (default: 1)

        Returns
        -------

        """
        for i in range(self.depth):
            index = self._hash(key, i)
            self.table[i][index] += count

    def estimate(self, key: bytes) -> int:
        """
        Estimate the count of the given key using the Count-Min Sketch.

        Parameters
        ----------
        key : bytes

        Returns
        -------
        int
        """
        min_estimate = float("inf")

        for i in range(self.depth):
            index = self._hash(key, i)
            min_estimate = min(min_estimate, self.table[i][index])

        return min_estimate

    def _hash(self, key: bytes, i: int) -> int:
        """
        Hash the key using the i-th hash function.

        Parameters
        ----------
        key : bytes
            The input key to hash
        i : int
            The index of the hash function

        Returns
        -------
        int
            The hashed index
        """
        # Hash the key using mmh3 with different seed values
        return mmh3.hash(key, self.hash_seeds[i]) % self.width
