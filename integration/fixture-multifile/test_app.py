import unittest

from app import greeting


class GreetingTests(unittest.TestCase):
    def test_normal_greeting(self):
        self.assertEqual(greeting("Ada"), "Hello, Ada.")

    def test_excited_greeting(self):
        self.assertEqual(greeting("Ada", excited=True), "Hello, Ada!")


if __name__ == "__main__":
    unittest.main()
