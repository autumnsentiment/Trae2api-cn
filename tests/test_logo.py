"""Project logo is served as the favicon and used in the console brand mark."""

from __future__ import annotations

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src import main as main_module


class LogoFaviconTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main_module.app)

    def test_favicon_routes_serve_svg_without_auth(self):
        for path in ("/favicon.svg", "/favicon.ico"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertTrue(response.headers["content-type"].startswith("image/svg+xml"))
            self.assertIn("<svg", response.text)
            self.assertIn("#22c55e", response.text)

    def test_console_links_favicon_and_uses_logo(self):
        html = self.client.get("/web/login").text
        self.assertIn('rel="icon" type="image/svg+xml" href="/favicon.svg"', html)
        brand = html.split('class="brand-mark">', 1)[1].split("</span>", 1)[0]
        self.assertIn('viewBox="0 0 64 64"', brand)

    def test_docs_logo_matches_served_logo(self):
        doc = Path(__file__).resolve().parent.parent / "docs" / "logo.svg"
        text = doc.read_text("utf-8")
        for fragment in ('M17 22h27', 'M39 14l9 8-9 8', 'M28 22v24', 'r="4.5"'):
            self.assertIn(fragment, text)
            self.assertIn(fragment, main_module.LOGO_SVG)


if __name__ == "__main__":
    unittest.main()
