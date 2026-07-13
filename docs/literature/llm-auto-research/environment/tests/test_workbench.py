import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ENV_DIR = Path(__file__).resolve().parents[1]


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, ENV_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CorpusWorkbenchTests(unittest.TestCase):
    def test_build_records_extracts_html_and_chunks_text(self):
        lib_corpus = load_module("lib_corpus")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "papers").mkdir()
            (root / "metadata").mkdir()
            (root / "papers" / "paper.html").write_text(
                "<html><body><h1>Demo</h1><p>Temporal coherence matters for AI Scientist research agents.</p></body></html>",
                encoding="utf-8",
            )
            (root / "metadata" / "papers_manifest.json").write_text(
                json.dumps(
                    [
                        {
                            "id": "01",
                            "key": "demo",
                            "title": "Demo Paper",
                            "year": "2026",
                            "category": "test",
                            "filename": "paper.html",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            records = lib_corpus.build_records(root, max_chars=60, overlap=10)

        self.assertGreaterEqual(len(records), 1)
        self.assertEqual(records[0]["paper_id"], "01")
        self.assertEqual(records[0]["title"], "Demo Paper")
        self.assertIn("Temporal coherence", " ".join(r["text"] for r in records))

    def test_search_records_ranks_query_overlap(self):
        search_papers = load_module("search_papers")
        records = [
            {
                "paper_id": "01",
                "title": "AI Scientist",
                "chunk_id": 0,
                "text": "autonomous research agent experiment loop novelty",
            },
            {
                "paper_id": "02",
                "title": "Video Diffusion",
                "chunk_id": 0,
                "text": "temporal coherence diffusion video generation",
            },
        ]

        results = search_papers.search_records("research agent novelty", records, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "AI Scientist")
        self.assertGreater(results[0]["score"], 0)

    def test_format_result_preserves_unicode_snippet(self):
        search_papers = load_module("search_papers")
        formatted = search_papers.format_result(
            {
                "score": 1.25,
                "paper_id": "01",
                "title": "Demo",
                "chunk_id": 0,
                "snippet": "novelty • verification",
            }
        )

        self.assertIn("novelty • verification", formatted)


if __name__ == "__main__":
    unittest.main()
