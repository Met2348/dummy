# Public Fallback Sources

This folder contains public, non-official fallback sources used when official
conference pages do not expose directly downloadable slides, videos, or reading
lists.

Use order:

1. Official keynote/invited-talk page in `../source-pages/main/`.
2. Official workshop page in `../source-pages/workshops/`.
3. Public GitHub/community index in this folder.
4. Search a speaker's homepage, lab page, or OpenReview/ACL Anthology profile.

Do not treat community lists as authoritative conference records. They are
second-entry maps for finding public slides, tutorials, paper trails, and
speaker-curated resources.

## Downloaded Community Sources

| Source | Why useful | Local snapshot |
|---|---|---|
| qingsongedu/awesome-AI-tutorials-surveys | Cross-conference tutorial, survey, keynote, and invited-talk index across AI venues | `source-pages/awesome_ai_tutorials_surveys.md` |
| qingsongedu/Talks-Keynotes-slides | Public talk/keynote/slide collection; useful as a downloadable-material fallback | `source-pages/talks_keynotes_slides.md` |
| soulbliss/NLP-conference-compendium | Community NLP conference resource compendium | `source-pages/nlp_conference_compendium.md` |
| AmberLJC/LLMSys-PaperList | LLM systems paper trail for ICLR/ICML/NeurIPS-style systems topics | `source-pages/llmsys_paperlist.md` |
| zhijing-jin/CausalNLP_Papers | Causal NLP paper map for robustness, reasoning, and evaluation topics | `source-pages/causalnlp_papers.md` |
| zhijing-jin/NLP4SocialGood_Papers | NLP social-good map for responsible, multilingual, and societal NLP topics | `source-pages/nlp4socialgood_papers.md` |
| heathersherry/Knowledge-Graph-Tutorials-and-Papers | Knowledge graph, retrieval, and reasoning background trail | `source-pages/knowledge_graph_tutorials_papers.md` |
| FeijiangHan/Top-Conference-Best-Papers | Best-paper filter to reduce random accepted-paper noise | `source-pages/top_conference_best_papers.md` |
| ivan-bilan/The-NLP-Pandect | Broad NLP learning index for filling background gaps | `source-pages/the_nlp_pandect.md` |
| FeiLiu36/LLM4Opt | LLM optimization/operations paper and tutorial trail | `source-pages/llm4opt.md` |

Manifest: `community_sources.csv`

Download log: `community_download_log.json`

## Notes On Missing Mirrors

`CoLM 2025 Visions of Language Modeling` remains URL-only. The official public
page is listed in `community_sources.csv`, but the local HTML snapshot was not
kept because Windows Defender flagged it. A public GitHub mirror was not found
at collection time, so the safe action is to revisit the official URL manually.
