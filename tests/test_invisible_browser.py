import base64
import json
import sys
import types
import unittest
from unittest.mock import patch

from orchestrator_v2.invisible_browser import run_probe


class FakePage:
    url = 'https://example.com/'

    def goto(self, url, **_kwargs):
        self.url = url

    def wait_for_load_state(self, *_args, **_kwargs):
        return None

    def title(self):
        return 'Example'

    def evaluate(self, expression):
        return {'document.readyState': 'complete', 'document.body ? document.body.innerText : \'\'': 'hello  world', 'navigator.webdriver': False}[expression]

    def screenshot(self, path, **_kwargs):
        with open(path, 'wb') as handle:
            handle.write(b'jpeg-image')

    def locator(self, _selector):
        return self

    def text_content(self):
        return 'quoted text'


class FakeBrowser:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def new_page(self):
        return FakePage()


class InvisibleBrowserTests(unittest.TestCase):
    def test_probe_is_bounded_and_returns_attachment_bytes(self):
        module = types.SimpleNamespace(InvisiblePlaywright=FakeBrowser)
        with patch.dict(sys.modules, {'invisible_playwright': module}), patch('orchestrator_v2.invisible_browser.socket.getaddrinfo', return_value=[(None, None, None, None, ('93.184.216.34', 443))]):
            result = run_probe({'url': 'https://example.com', 'selector': 'h1', 'screenshot': True})
        self.assertEqual(result['status'], 'OK')
        self.assertEqual(result['body_text'], 'hello world')
        self.assertNotIn('password', json.dumps(result).lower())
        self.assertEqual(base64.b64decode(result['screenshot_base64']), b'jpeg-image')

    def test_secondary_navigation_is_same_origin(self):
        from orchestrator_v2.invisible_browser import _public_url
        with patch('orchestrator_v2.invisible_browser.socket.getaddrinfo', return_value=[(None, None, None, None, ('93.184.216.34', 443))]):
            with self.assertRaises(ValueError):
                _public_url('https://other.example/', same_origin='https://example.com/')

    def test_private_targets_are_rejected(self):
        from orchestrator_v2.invisible_browser import _public_url
        with patch('orchestrator_v2.invisible_browser.socket.getaddrinfo', return_value=[(None, None, None, None, ('127.0.0.1', 80))]):
            with self.assertRaises(ValueError):
                _public_url('http://localhost/')


if __name__ == '__main__':
    unittest.main()

