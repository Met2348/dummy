# Local Research Workbench Deployment

This is the first local smoke-test environment for the LLM Auto Research literature package.

## Goal

Turn the 70 local papers into a searchable local corpus:

```text
papers/ + papers_manifest.json -> corpus.jsonl -> local search
```

This first version intentionally does not require an API key, Ollama, Chroma, or a GPU.

## Setup

```powershell
python -m pip install -r docs/literature/llm-auto-research/environment/requirements.txt
```

## Build Corpus

```powershell
python docs/literature/llm-auto-research/environment/build_corpus.py
```

Output:

```text
docs/literature/llm-auto-research/metadata/corpus.jsonl
```

## Search Locally

```powershell
python docs/literature/llm-auto-research/environment/search_papers.py "novelty verification ai scientist" --top-k 5
```

## Verify

```powershell
python docs/literature/llm-auto-research/environment/verify_env.py
python -m unittest discover docs/literature/llm-auto-research/environment/tests -v
```

## Next Layer

After this smoke test works, the next useful layer is a persistent vector index and a small `paper -> claim -> evidence -> gap` card generator.
