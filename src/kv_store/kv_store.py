import threading
import time


class SimpleKeyValueStore:
    """A simple in-memory key-value store implementation using a dictionary."""

    def __init__(self):
        """Initialize an empty key-value store."""
        self.store = {}
        self.lock = threading.Lock()

    def set(self, key, value):
        """Store a value associated with the given key.

        Args:
            key: The key to associate with the value.
            value: The value to store.
        """
        with self.lock:
            self.store[key] = value

    def get(self, key):
        """Retrieve the value associated with the given key.

        Args:
            key: The key to look up.

        Returns:
            The value associated with the key, or None if the key doesn't exist.
        """
        return self.store.get(key)

    def delete(self, key):
        """Remove the key and its associated value from the store.

        Args:
            key: The key to delete.
        """
        if key in self.store:
            del self.store[key]

    def keys(self):
        """Return a list of all keys in the store.

        Returns:
            A list containing all keys currently stored.
        """
        return list(self.store.keys())


def test_stress():
    store = SimpleKeyValueStore()
    num_threads = 50
    operations = 10000

    def worker(thread_id):
        for i in range(operations):
            store.set(f"thread_{thread_id}_key_{i}", i)
            store.get(f"thread_{thread_id}_key_{i}")
            if i % 2 == 0:
                store.delete(f"thread_{thread_id}_key_{i}")

    start = time.time()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    end = time.time()

    print(f"Completed {num_threads * operations} operations in {end - start:.2f}s")


if __name__ == "__main__":
    test_stress()
