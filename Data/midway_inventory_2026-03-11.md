# KLab Midway Directory Inventory

**Location:** /project/jevans on Midway3
**Scan Date:** 2026-03-11
**Last Updated:** 2026-03-11T02:00:06.685234
**Total Resources:** 133 directories

This document contains an inventory of all directories and datasets available
on Midway3 in the /project/jevans space. Use this to find data, identify owners,
and understand what resources are available.

## Quick Reference - What Data is Available?

The following types of data are available on Midway:

**Bibliographic Database:** semantic_scholar_Jan31_2025, S2AG_Dec_2024_snapshot, openalex-snapshot, dimensions

**Bibliographic Database (Historical):** MAG_Dec_2021_snapshot

**Biomedical Citation Metrics:** iCite_bulk_snapshot_Sep_2023

**Biomedical Knowledge Graph:** PKG_v2_2023

**Chemical Database:** pubchem_compound_2025_May

**Computer Science Bibliography:** DBLP_2025_Jan

**Drug Database:** drugbank_5.1.13

**ML Benchmark Data:** PwC_Mar_04_2025, PwC_Nov_13_2024

**Other Data:** corpora

**Patent Database:** PATSTAT

**Preprint Server Data:** ssrn_html_data

**Social Media Data:** reddit_data

### Common Search Terms

This inventory includes data related to: publications, papers, citations, academic literature,
bibliographic databases, preprints, pre-prints, arxiv, biorxiv, patents, patent data, USPTO,
EPO, PATSTAT, social media, Reddit, web crawl, machine learning, ML models, biomedical,
PubMed, NIH, grants, clinical trials, knowledge graphs, embeddings, research data, datasets.

---

## Datasets on Midway

Shared datasets available for research. Organized by category.

### Bibliographic Database

### semantic_scholar_Jan31_2025
- **Also known as:** S2AG, S2, SemanticScholar, Semantic Scholar
- **Path:** `/project/jevans/semantic_scholar_Jan31_2025`
- **Owner:** kangd
- **Size:** 39 GB
- **Category:** Bibliographic Database
- **Type:** Major Data Snapshot Directory
- **Contains:** paper metadata, author data, citation links
- **Status:** VALUABLE, ACTIVE
- **Description:** **VALUABLE DATA** - Even more recent Semantic Scholar snapshot       (January 31, 2025). The 21GB comprehensive file covers paper titles,       venues, and years across all publications. The 18GB author linkage file       provides complete authorship data. The separate 2024 subset enables       studying very recent publications. Multiple processing notebooks show       active use and methodology development. This is the most current       Semantic Scholar data in the lab (newer than S2AG_Dec_202
- **Related terms:** abstracts, academic literature, bibliography, citation network, citations, papers, publications, research papers, scientific papers
- **Last Modified:** 2025-06-11

### S2AG_Dec_2024_snapshot
- **Also known as:** S2AG, S2, SemanticScholar, Semantic Scholar
- **Path:** `/project/jevans/S2AG_Dec_2024_snapshot`
- **Owner:** jacdals
- **Size:** 50 GB
- **Category:** Bibliographic Database
- **Type:** Major Data Snapshot Directory
- **Contains:** paper metadata, author data, citation links
- **Status:** VALUABLE, ACTIVE
- **Description:** **VALUABLE DATA** - Semantic Scholar Academic Graph December       2024 snapshot. S2AG is a major open bibliographic database with paper       metadata, abstracts, citations, and author information across all fields.       The inclusion of S2ORC data provides full text and structured paper       content. Recent snapshot (December 2024) means current coverage. The 33GB       papers file contains comprehensive metadata. Processing scripts show       active use with parallel extraction for scalabil
- **Related terms:** abstracts, academic literature, bibliography, citation network, citations, papers, publications, research papers, scientific papers
- **Last Modified:** 2025-08-27

### openalex-snapshot
- **Also known as:** Microsoft Academic Graph, OpenAlex, MAG, OA, Microsoft Academic
- **Path:** `/project/jevans/openalex-snapshot`
- **Owner:** beichenlu
- **Size:** Unknown
- **Category:** Bibliographic Database
- **Type:** Major Data Snapshot Directory
- **Contains:** paper metadata, author data, citation links, institution data
- **Status:** VALUABLE, ACTIVE
- **Description:** **VALUABLE DATA** - OpenAlex bibliographic database snapshot.       OpenAlex is a comprehensive open-access replacement for Microsoft Academic       Graph covering publications, authors, institutions, venues, and concepts       across all fields. Recent snapshot (December 2024) means current data.       The 14 subdirectories suggest partitioned data for distributed processing.       Highly valuable for any bibliometric research. OpenAlex provides clean,       structured data with DOIs, citations
- **Related terms:** MAG replacement, academic literature, bibliography, citations, discontinued 2022, open access, papers, publications, research papers, scientific papers
- **Last Modified:** 2025-12-19

### dimensions
- **Also known as:** Dimensions
- **Path:** `/project/jevans/dimensions`
- **Owner:** akozlo
- **Size:** Unknown
- **Category:** Bibliographic Database
- **Type:** Major Data Snapshot Directory
- **Contains:** paper metadata, patent links, grant data, citation data
- **Status:** VALUABLE, ACTIVE
- **Description:** **VALUABLE DATA** - Recent (June 2025) snapshot of the       Dimensions bibliographic database. Dimensions is a major commercial       research database covering publications, citations, grants, patents,       clinical trials, and policy documents. This is enterprise-grade       bibliometric data with broader coverage than many academic-only databases.       The GCP script suggests cloud-based processing infrastructure. Highly       valuable for comprehensive bibliometric analysis, though requir
- **Related terms:** citations, clinical trials, commercial database, comprehensive bibliometrics, grants, papers, patents, policy documents, publications
- **Last Modified:** 2025-06-12

### Bibliographic Database (Historical)

### MAG_Dec_2021_snapshot
- **Also known as:** Microsoft Academic, MAG, Microsoft Academic Graph
- **Path:** `/project/jevans/MAG_Dec_2021_snapshot`
- **Owner:** kangd
- **Size:** Unknown
- **Category:** Bibliographic Database (Historical)
- **Type:** Major Data Snapshot Directory
- **Contains:** paper metadata, author data, citation links
- **Status:** VALUABLE
- **Description:** **VALUABLE DATA** - December 2021 snapshot of the Microsoft       Academic Graph (MAG), one of the largest academic publication databases       before it was discontinued in 2022. This is a historical snapshot of       significant value since MAG is no longer updated. Contains comprehensive       publication, author, institution, and citation data across all scientific       fields. The presence of specialized subdirectories (nlp, samples, advanced)       suggests this has been processed and org
- **Related terms:** academic literature, bibliography, citations, discontinued 2022, papers, publications, research papers, scientific papers
- **Last Modified:** 2022-01-04

### Computer Science Bibliography

### DBLP_2025_Jan
- **Also known as:** dblp, DBLP
- **Path:** `/project/jevans/DBLP_2025_Jan`
- **Owner:** kangd
- **Size:** 10 GB
- **Category:** Computer Science Bibliography
- **Type:** Data Snapshot Directory
- **Contains:** paper metadata, author data, venue data
- **Status:** VALUABLE, ACTIVE
- **Description:** **VALUABLE DATA** - Recent (January 2025) snapshot of the       DBLP computer science bibliography database. Provided in both XML and       JSON Lines format for flexibility. DBLP is a comprehensive bibliography       of computer science publications. This is high-quality, well-structured       bibliometric data valuable for any computer science publication analysis,       citation networks, author collaboration studies, or venue analysis.       Recent snapshot means current data. Parser include
- **Related terms:** CS papers, author collaboration, bibliography, computer science, conferences, journals, papers, publications, venues
- **Last Modified:** 2025-01-28

### Patent Database

### PATSTAT
- **Also known as:** EPO PATSTAT, PATSTAT
- **Path:** `/project/jevans/PATSTAT`
- **Owner:** lkcao
- **Size:** 64 GB
- **Category:** Patent Database
- **Type:** Major Data Snapshot Directory
- **Contains:** legal status, patent citations, patent bibliographic data
- **Status:** VALUABLE, ACTIVE
- **Description:** **VALUABLE DATA** - PATSTAT (Patent Statistical Database)       from EPO (European Patent Office). Autumn 2023 edition provides       comprehensive global patent data including bibliographic information,       citations, classifications, and legal status. PATSTAT is the gold       standard for patent research with clean, structured data covering       worldwide patents. The 11-part structure enables distributed processing.       OECD complementary datasets add economic indicators. Highly valuabl
- **Related terms:** EPO, European Patent Office, IP data, assignees, intellectual property, inventors, patent citations, patent classifications, patent data, patent families, patents
- **Last Modified:** 2025-03-26

### Preprint Server Data

### ssrn_html_data
- **Also known as:** SSRN, ssrn
- **Path:** `/project/jevans/ssrn_html_data`
- **Owner:** amritap1
- **Size:** 0.0 MB
- **Category:** Preprint Server Data
- **Contains:** HTML data, abstracts, preprint metadata
- **Related terms:** SSRN papers, economics papers, law papers, pre-prints, preprints, social science preprints, working papers
- **Last Modified:** 2024-11-19

### Biomedical Citation Metrics

### iCite_bulk_snapshot_Sep_2023
- **Also known as:** iCite, NIH iCite
- **Path:** `/project/jevans/iCite_bulk_snapshot_Sep_2023`
- **Owner:** kangd
- **Size:** 41 GB
- **Category:** Biomedical Citation Metrics
- **Type:** Data Snapshot Directory
- **Contains:** grant linkages, impact scores, citation metrics
- **Status:** VALUABLE
- **Description:** **VALUABLE DATA** - Major dataset from NIH's iCite system       tracking citations, bibliometrics, and impact metrics for NIH-funded       research. iCite provides Relative Citation Ratios (RCR), field-normalized       citation metrics, and NIH grant linkages. The open citation collection       enables citation network analysis. Snapshot from September 2023 provides       historical baseline. Highly valuable for studying biomedical research       impact, NIH funding effectiveness, and citation p
- **Related terms:** NIH, NIH funding, RCR, bibliometrics, biomedical, citations, grant linkages, impact metrics, relative citation ratio
- **Last Modified:** 2023-11-06

### Biomedical Knowledge Graph

### PKG_v2_2023
- **Also known as:** PKG, PubMed Knowledge Graph
- **Path:** `/project/jevans/PKG_v2_2023`
- **Owner:** kangd
- **Size:** 144 GB
- **Category:** Biomedical Knowledge Graph
- **Type:** Major Data Snapshot Directory
- **Contains:** entity linkages, grant-paper links, knowledge graph
- **Status:** VALUABLE
- **Description:** **VALUABLE DATA - COMPREHENSIVE BIOMEDICAL KNOWLEDGE GRAPH** -       PubMed Knowledge Graph integrates PubMed articles with NIH grants,       clinical trials, patents, MeSH terms, and biological entities. The 144GB       dataset provides unprecedented linkages across biomedical research       outputs. Particularly valuable features: grant-paper linkages (A05),       bioentity relationships (C21), clinical trial connections (C11-C14),       and patent-paper links (C16-C18). The author-inventor li
- **Related terms:** MeSH, NIH, PubMed, bioentities, biomedical, clinical trials, drug-paper links, gene-paper links, grants, knowledge graph
- **Last Modified:** 2025-06-16

### Chemical Database

### pubchem_compound_2025_May
- **Also known as:** PubChem
- **Path:** `/project/jevans/pubchem_compound_2025_May`
- **Owner:** kangd
- **Size:** 0.0 MB
- **Category:** Chemical Database
- **Contains:** compound data, chemical structures
- **Related terms:** bioassays, chemical data, chemical structures, chemicals, compounds, drug compounds, molecules
- **Last Modified:** 2025-05-27

### Drug Database

### drugbank_5.1.13
- **Also known as:** DrugBank
- **Path:** `/project/jevans/drugbank_5.1.13`
- **Owner:** kangd
- **Size:** 0.0 MB
- **Category:** Drug Database
- **Contains:** drug-target interactions, drug metadata
- **Related terms:** drug data, drug interactions, drug targets, drugs, medications, pharmaceuticals, pharmacology
- **Last Modified:** 2025-06-13

### Social Media Data

### reddit_data
- **Also known as:** Reddit
- **Path:** `/project/jevans/reddit_data`
- **Owner:** rnhe
- **Size:** 0.0 MB
- **Category:** Social Media Data
- **Contains:** posts, comments, user data
- **Related terms:** Reddit, comments, online communities, posts, social media, subreddits, text data, user discussions
- **Last Modified:** 2026-01-14

### ML Benchmark Data

### PwC_Mar_04_2025
- **Also known as:** Papers with Code, PapersWithCode, PwC
- **Path:** `/project/jevans/PwC_Mar_04_2025`
- **Owner:** kangd
- **Size:** 0.0 MB
- **Category:** ML Benchmark Data
- **Contains:** benchmark data, paper-code links
- **Related terms:** AI papers, ML papers, benchmarks, code, datasets, leaderboards, machine learning, models, research papers
- **Last Modified:** 2025-03-04

### PwC_Nov_13_2024
- **Also known as:** Papers with Code, PapersWithCode, PwC
- **Path:** `/project/jevans/PwC_Nov_13_2024`
- **Owner:** kangd
- **Size:** 0.0 MB
- **Category:** ML Benchmark Data
- **Contains:** benchmark data, paper-code links
- **Related terms:** AI papers, ML papers, benchmarks, code, datasets, leaderboards, machine learning, models, research papers
- **Last Modified:** 2025-03-04

### Other Datasets

### corpora
- **Path:** `/project/jevans/corpora`
- **Owner:** 1608369274
- **Size:** 0.0 MB
- **Last Modified:** 2021-09-06

## Project Directories

Shared project directories with collaborative research work.

### apto_data_engineering
- **Also known as:** US Patents, USPTO
- **Path:** `/project/jevans/apto_data_engineering`
- **Owner:** nadavkunievsky
- **Size:** 17 GB
- **Category:** Patent Database
- **Type:** Shared Project Directory - Data Engineering
- **Contains:** patent embeddings, patent metadata
- **Status:** VALUABLE, ACTIVE
- **Description:** **VALUABLE DATA AND ACTIVE PROJECT** - Major data       engineering initiative aggregating multiple scientific preprint servers,       patent data, and academic literature. Well-organized with clear       documentation. Contains substantial processed datasets including USPTO       patent novelty embeddings spanning decades. The personal folders suggest       active multi-researcher collaboration. Code appears clean and functional.       This is a central data repository for multiple lab projects
- **Related terms:** US patents, intellectual property, inventors, patent data, patent embeddings, patent novelty, patents
- **Last Modified:** 2025-09-30

### interp_group
- **Path:** `/project/jevans/interp_group`
- **Owner:** akozlo
- **Size:** 50-100 MB
- **Category:** ML Models
- **Type:** Shared Project Directory - AI Interpretability
- **Contains:** model configs, model weights
- **Status:** VALUABLE, ACTIVE
- **Description:** **ACTIVE RESEARCH** - Collaborative group working on AI       interpretability and mechanistic understanding of language models. Very       recent activity (January 2026). The prompt builder and vector pipeline       notebooks suggest research on how LLMs represent and process information.       Well-organized collaboration structure with personal workspaces. Valuable       for understanding current approaches to AI interpretability research. The       group structure indicates this is an ongoin
- **Related terms:** ML models, embeddings, machine learning, model weights, neural networks, pretrained models, transformers
- **Last Modified:** 2026-03-06

### gender_surprise
- **Also known as:** OA, OpenAlex
- **Path:** `/project/jevans/gender_surprise`
- **Owner:** jlockhart
- **Size:** 8 GB
- **Category:** Bibliographic Database
- **Type:** Shared Project Directory
- **Contains:** paper metadata, author data, citation links, institution data
- **Status:** VALUABLE, ACTIVE
- **Description:** **VALUABLE DATA** - Comprehensive author gender inference       dataset covering both OpenAlex and Web of Science. Contains processed       name-to-gender mappings for US and international authors with both first       and middle name analysis. The large pickle files suggest extensive       preprocessing of bibliometric data. This is valuable for gender bias       research, authorship pattern analysis, and demographic studies in science.       Well-organized with clear naming conventions. Recent
- **Related terms:** MAG replacement, academic literature, bibliography, citations, open access, papers, publications, research papers, scientific papers
- **Last Modified:** 2025-11-22

### tip
- **Also known as:** OA, OpenAlex
- **Path:** `/project/jevans/tip`
- **Owner:** jlockhart
- **Size:** 1-10 GB
- **Category:** Bibliographic Database
- **Type:** Major Project Directory
- **Contains:** citation links, model configs, institution data, model weights, paper metadata, author data
- **Status:** VALUABLE, ACTIVE
- **Description:** **MAJOR PROJECT** - The "tip" (possibly "topics in papers"       or similar) project appears to be a major research effort on modeling       scientific topics and their evolution. Multiple model versions (1, 11, 12)       show iterative development. The progression from MEDLINE to OpenAlex       suggests expanding scope. Network modeling and disruption analysis       indicate focus on innovation and scientific change. The "prescience       graveyard" shows methodical experimentation. Recent acti
- **Related terms:** MAG replacement, ML models, academic literature, bibliography, citations, embeddings, machine learning, model weights, neural networks, open access, papers, pretrained models, publications, research papers, scientific papers
- **Last Modified:** 2025-04-16

### agent_42_pilot
- **Path:** `/project/jevans/agent_42_pilot`
- **Owner:** giochoi
- **Size:** 33 MB
- **Type:** Project Directory
- **Description:** Small experimental project, likely exploratory. No significant       data storage. Appears to be agent-based simulation or AI agent work.
- **Last Modified:** 2025-04-30

### crystal_embedding
- **Path:** `/project/jevans/crystal_embedding`
- **Owner:** jsourati
- **Size:** <10 MB
- **Type:** Small Project Directory
- **Description:** Small project from 2023 on crystal embeddings (materials       science?). Minimal activity, appears to be a proof-of-concept or       exploratory project. Limited value for reuse given age and small scope.
- **Last Modified:** 2025-04-30

### subspace_sos
- **Path:** `/project/jevans/subspace_sos`
- **Owner:** shiyanglai
- **Size:** 0.0 MB
- **Last Modified:** 2024-01-10

### QuestionAnswerApproach
- **Path:** `/project/jevans/QuestionAnswerApproach`
- **Owner:** nadavkunievsky
- **Size:** 0.0 MB
- **Last Modified:** 2025-03-27

### llm_social_simul
- **Path:** `/project/jevans/llm_social_simul`
- **Owner:** junsol
- **Size:** 0.0 MB
- **Last Modified:** 2023-05-16

### Wikipedia_rp
- **Path:** `/project/jevans/Wikipedia_rp`
- **Owner:** lkcao
- **Size:** 0.0 MB
- **Last Modified:** 2025-12-06

## Researcher Directories

Individual researcher directories. Those marked VALUABLE contain significant data.

### akozlo
- **Also known as:** CommonCrawl, Common Crawl
- **Path:** `/project/jevans/akozlo`
- **Owner:** akozlo
- **Size:** 214 GB
- **Category:** Web Crawl Data
- **Type:** Individual Researcher Directory
- **Contains:** HTML, text, model weights, web pages, model configs
- **Status:** VALUABLE, ACTIVE
- **Description:** **VALUABLE DATA AND ACTIVE RESEARCH** - This is a major       active research directory with 214GB of data. The "digital_doubles"       project appears to be using Common Crawl web data and transformer models       to create digital representations of people or personas. Multiple       collaborators and structured RA work suggests this is a well-organized,       ongoing project. The size suggests substantial raw data and model       artifacts are stored here.
- **Related terms:** HTML data, ML models, embeddings, internet archive, machine learning, model weights, neural networks, pretrained models, transformers, web corpus, web crawl, web data, web pages, web text
- **Last Modified:** 2026-02-13

### adarshm
- **Also known as:** Telegram, Reddit
- **Path:** `/project/jevans/adarshm`
- **Owner:** adarshm
- **Size:** 76 GB
- **Category:** Social Media Data
- **Type:** Individual Researcher Directory
- **Contains:** comments, user data, channel data, messages, posts
- **Status:** VALUABLE
- **Description:** **VALUABLE DATA** - 76GB of social media data (Reddit and       Telegram). The "ideolect" project appears to be analyzing linguistic       patterns in social media. PostgreSQL database infrastructure suggests       this is structured, queryable data. Code is utility-level for data       management rather than novel analysis.
- **Related terms:** Reddit, Telegram, channels, chat data, comments, groups, messaging, online communities, posts, social media, subreddits, text data, user discussions
- **Last Modified:** 2022-02-27

### gio
- **Path:** `/project/jevans/gio`
- **Owner:** giochoi
- **Size:** 100-500 MB
- **Type:** Individual Researcher Directory - Multiple Projects
- **Status:** VALUABLE, ACTIVE
- **Description:** **ACTIVE RESEARCH** - Diverse portfolio focused on scientific       integrity ("dishonesty"), publication patterns ("bready"), and causal       methods. Very recent activity (2025). The dishonesty project appears to       be the major focus with substantial scale. Well-organized project       structure with clear separation of concerns. Multiple methodological       projects (autogen, causal_inference, hypotheses) suggest development of       reusable research tools. Valuable for understanding r
- **Last Modified:** 2026-02-04

### haiziyu
- **Also known as:** Microsoft Academic, MAG, Microsoft Academic Graph
- **Path:** `/project/jevans/haiziyu`
- **Owner:** haiziyu
- **Size:** 122 GB
- **Category:** Bibliographic Database (Historical)
- **Type:** Individual Researcher Directory - PMI Computations
- **Contains:** paper metadata, author data, citation links
- **Status:** VALUABLE
- **Description:** **VALUABLE DATA - MAJOR COMPUTATION** - Massive pre-computed       PMI (word co-occurrence) statistics for scientific abstracts spanning       over a century. The 48GB paper-word mapping and extensive temporal bins       enable longitudinal analysis of scientific language evolution, concept       emergence, and semantic change. This represents very expensive computation       on the full MAG corpus. Highly valuable for computational linguistics,       science of science, and historical analysis
- **Related terms:** academic literature, bibliography, citations, discontinued 2022, papers, publications, research papers, scientific papers
- **Last Modified:** 2022-09-11

### shiyang
- **Path:** `/project/jevans/shiyang`
- **Owner:** shiyanglai
- **Size:** 1-5 GB
- **Category:** ML Models
- **Type:** Individual Researcher Directory
- **Contains:** model configs, model weights
- **Status:** VALUABLE, ACTIVE
- **Description:** **ACTIVE RESEARCH** - Diverse portfolio spanning AI bias,       computational social science, and neural network interpretability. The       superposition-neurons project suggests cutting-edge work on how neural       networks represent information. Recent activity (February 2025). Multiple       projects on AI's societal implications show engagement with responsible       AI research. The large models directory (29 subdirs) indicates substantial       computational work. Valuable for AI safety,
- **Related terms:** ML models, embeddings, machine learning, model weights, neural networks, pretrained models, transformers
- **Last Modified:** 2026-03-01

### maxzhuyt
- **Path:** `/project/jevans/maxzhuyt`
- **Owner:** maxzhuyt
- **Size:** 500 MB - 1 GB
- **Category:** ML Models
- **Type:** Individual Researcher Directory - Political Science
- **Contains:** model configs, model weights
- **Status:** VALUABLE, ACTIVE
- **Description:** **ACTIVE RESEARCH** - Political science work using neural       network interpretability methods to analyze political ideology. Combines       major political science datasets (ANES, CHES, Manifesto Project) with       deep learning analysis (PCA on activations, layer analysis). Very recent       activity (September 2025). The activation caching and layer analysis       suggest mechanistic interpretability work on how models represent political       concepts. High-quality visualizations indicat
- **Related terms:** ML models, embeddings, machine learning, model weights, neural networks, pretrained models, transformers
- **Last Modified:** 2026-03-07

### likun
- **Path:** `/project/jevans/likun`
- **Owner:** lkcao
- **Size:** 3-5 GB
- **Type:** Individual Researcher Directory
- **Status:** VALUABLE, ACTIVE
- **Description:** **ACTIVE RESEARCH** - Focus on cognitive science and machine       learning, particularly studying depth vs. breadth tradeoffs and emergence       phenomena. The large emergence project (32K files) appears to be a major       focus. Recent activity (2025). Work combines theoretical analysis (RTF       documents) with computational experiments. The cognitive visualization       project suggests interdisciplinary work bridging AI and cognitive science.       The depth/breadth retraining work is re
- **Last Modified:** 2026-02-23

### ruininghe
- **Path:** `/project/jevans/ruininghe`
- **Owner:** rnhe
- **Size:** 500 MB - 2 GB
- **Category:** ML Models
- **Type:** Individual Researcher Directory
- **Contains:** model configs, model weights
- **Status:** VALUABLE, ACTIVE
- **Description:** **ACTIVE RESEARCH** - Machine learning project with       comprehensive structure (code, configs, data, models). The computed       models directory with 8K files suggests extensive experimentation.       High-quality visualizations indicate publication-ready analysis. Binary       target analysis suggests classification research. Recent activity (2025).       Well-organized project with clear separation of raw data, computed results,       and models. Valuable for understanding ML experimentati
- **Related terms:** ML models, embeddings, machine learning, model weights, neural networks, pretrained models, transformers
- **Last Modified:** 2026-02-17

### Honglin_Bao
- **Path:** `/project/jevans/Honglin_Bao`
- **Owner:** honglinbao
- **Size:** 50-200 MB
- **Type:** Individual Researcher Directory - Multiple Projects
- **Status:** VALUABLE, ACTIVE
- **Description:** **ACTIVE RESEARCH** - Diverse portfolio spanning AI       methodology (chain-of-hints, LLM fine-tuning), scientific analysis       (hypothesis discovery, patents, dishonesty), and computational methods       (distance metrics, benchmarking). Very recent activity (2025). The       hypothesis discovery and attention inequality projects appear substantial.       Well-organized with clear project separation. High value for AI research       methodology and bibliometric analysis. Shows methodological
- **Last Modified:** 2026-01-11

### esposito
- **Path:** `/project/jevans/esposito`
- **Owner:** cresposito
- **Size:** 695 GB
- **Type:** Individual Researcher Directory - Major Computations
- **Status:** VALUABLE, ACTIVE
- **Description:** **VALUABLE DATA - EXTREMELY LARGE COMPUTATION** - Contains       massive pre-computed similarity matrices for patent descriptions and       summaries across multiple time periods. These are N×N matrices where N       is the number of patents (~millions), explaining the enormous file sizes.       This represents months/years of computation time on patent text similarity       that would be prohibitively expensive to reproduce. The data enables       research on patent substitution, technological
- **Last Modified:** 2025-03-17

### bernie
- **Also known as:** Reddit
- **Path:** `/project/jevans/bernie`
- **Owner:** bernardkoch
- **Size:** 150 MB
- **Category:** Social Media Data
- **Type:** Shared Project Directory
- **Contains:** posts, comments, user data
- **Status:** VALUABLE
- **Description:** **VALUABLE DATA** - Contains toxicity-scored datasets for       4chan and Reddit content. The polcleaner.py script is production-quality       code for cleaning messy social media CSV data. The use of Feather format       suggests performance-conscious data handling. This appears to be research       on online toxicity in controversial forums. Data and code could be useful       for future social media or content moderation research.
- **Related terms:** Reddit, comments, online communities, posts, social media, subreddits, text data, user discussions
- **Last Modified:** 2025-04-30

### Daniela
- **Also known as:** OA, OpenAlex
- **Path:** `/project/jevans/Daniela`
- **Owner:** danqingchen
- **Size:** 54 GB
- **Category:** Bibliographic Database
- **Type:** Individual Researcher Directory
- **Contains:** paper metadata, author data, citation links, institution data
- **Status:** VALUABLE, ACTIVE
- **Description:** **VALUABLE DATA AND ACTIVE RESEARCH** - Major project on       concept co-occurrence and author-concept relationships, likely using       OpenAlex or similar bibliometric data. Contains substantial processed       datasets with hierarchical concept classifications (levels 1 and 2).       The 14GB JSON file and large parquet directories suggest comprehensive       coverage. Very active recent work (2025). The concept pair analysis       could be highly valuable for bibliometric research, knowledg
- **Related terms:** MAG replacement, academic literature, bibliography, citations, open access, papers, publications, research papers, scientific papers
- **Last Modified:** 2025-05-22

### mschwarting
- **Path:** `/project/jevans/mschwarting`
- **Owner:** meschw04
- **Size:** 216 MB | Owner: meschw04
- **Type:** Results directory
- **Status:** VALUABLE, ACTIVE
- **Description:** Product factsheet data download. Duplicate/related to apto_data_engineering factsheet work. Moderate value.
- **Last Modified:** 2025-07-28

### jacdals
- **Path:** `/project/jevans/jacdals`
- **Owner:** jacdals
- **Size:** 16 GB
- **Type:** Individual Researcher Directory - Medical/Drug Research
- **Status:** VALUABLE, ACTIVE
- **Description:** **VALUABLE DATA AND ACTIVE RESEARCH** - Comprehensive       collection combining bibliometric data (WoS) with medical/pharmaceutical       databases (MeSH, ChEBI, WHO ATC). The 7GB author address file is       particularly valuable for geographic analysis of science. Multiple active       projects on funding inequality, FAIR data principles, and research       optimization. The conference bandits project suggests algorithmic       approaches to resource allocation. Well-organized with clean data
- **Last Modified:** 2026-02-16

### hongkai
- **Also known as:** Reddit
- **Path:** `/project/jevans/hongkai`
- **Owner:** hongkai
- **Size:** 50 GB
- **Category:** Social Media Data
- **Type:** Individual Researcher Directory
- **Contains:** posts, comments, user data
- **Status:** VALUABLE, ACTIVE
- **Description:** **VALUABLE DATA** - Extensive Reddit data collection with       millions of entries, organized with and without scores. Major focus on       LLM-based social simulation and stance detection. The large SLURM script       and multiprocessing code suggest serious computational work. Recent       activity (2024). The social simulation work using LLMs is cutting-edge       research. Valuable for social media analysis, computational social science,       and LLM evaluation on social tasks.
- **Related terms:** Reddit, comments, online communities, posts, social media, subreddits, text data, user discussions
- **Last Modified:** 2024-04-17

### nadav
- **Path:** `/project/jevans/nadav`
- **Owner:** nadavkunievsky
- **Size:** 1-5 GB
- **Type:** Individual Researcher Directory - Extensive Portfolio
- **Status:** VALUABLE, ACTIVE
- **Description:** **ACTIVE RESEARCH - HIGHLY PRODUCTIVE** - Exceptionally       diverse portfolio spanning AI methodology, economics, social science, and       bibliometrics. Very recent activity (January 2025). The 38 projects       indicate a highly productive researcher with interests in computational       social science, AI for scientific discovery, and economic analysis.       Multiple projects on funding and research evaluation suggest expertise in       science policy. The Info-Flows-LLMs and generating_s
- **Last Modified:** 2026-03-04

### nwrim
- **Path:** `/project/jevans/nwrim`
- **Owner:** nwrim
- **Size:** 175 GB
- **Category:** ML Models
- **Type:** Individual Researcher Directory - Psychology Data
- **Contains:** model configs, model weights
- **Status:** VALUABLE, ACTIVE
- **Description:** **VALUABLE DATA - MASSIVE PSYCHOLOGY DATASET** - Contains       175GB of compressed psychology data. The "perspectives" and "prior"       datasets suggest research on psychological priors and perspectives,       possibly related to belief formation or cognitive biases. The public       goods game (pgg) environment indicates experimental economics or       behavioral game theory research. The social technology demo suggests       computational modeling of social behavior. Recent compilation (Marc
- **Related terms:** ML models, embeddings, machine learning, model weights, neural networks, pretrained models, transformers
- **Last Modified:** 2025-03-29

### hyunkukwon
- **Also known as:** Reddit
- **Path:** `/project/jevans/hyunkukwon`
- **Owner:** hyunkukwon
- **Size:** 43 GB
- **Category:** Social Media Data
- **Type:** Individual Researcher Directory - Political Text Analysis
- **Contains:** posts, comments, user data
- **Status:** VALUABLE
- **Description:** **VALUABLE DATA** - Large-scale political text corpora with       conservative/liberal labels, including both human-written and GPT-generated       text. The balanced GPT corpora (21GB total) are particularly interesting       for studying political language and AI-generated political content. Data       appears to be from Reddit or similar social media ("austin" may refer to       r/Austin subreddit). Well-organized with multiple sample sizes for       different computational needs. From 2021-2
- **Related terms:** Reddit, comments, online communities, posts, social media, subreddits, text data, user discussions
- **Last Modified:** 2024-02-20

### Dawoon
- **Also known as:** OA, Dimensions, OpenAlex
- **Path:** `/project/jevans/Dawoon`
- **Owner:** jdwoon0523
- **Size:** 50-100 MB
- **Category:** Bibliographic Database
- **Type:** Individual Researcher Directory
- **Contains:** citation links, institution data, grant data, patent links, paper metadata, author data, citation data
- **Status:** VALUABLE, ACTIVE
- **Description:** **ACTIVE RESEARCH** - Diverse portfolio of projects focusing       on embedding spaces, curvature analysis, and multiple data domains       (phones, reviews, legal, scientific). Very recent activity (January 2026).       The embedding and curvature work suggests mathematical/geometric analysis       of representation spaces. Multiple data sources integrated (GSMArena,       Amazon, Dimensions, OpenAlex). Well-organized project structure. The       variety suggests either exploratory research or
- **Related terms:** MAG replacement, academic literature, bibliography, citations, clinical trials, commercial database, comprehensive bibliometrics, grants, open access, papers, patents, policy documents, publications, research papers, scientific papers
- **Last Modified:** 2026-03-09

### zhenzhang
- **Also known as:** OA, OpenAlex
- **Path:** `/project/jevans/zhenzhang`
- **Owner:** zhenzhang
- **Size:** 2-10 GB
- **Category:** Bibliographic Database
- **Type:** Individual Researcher Directory
- **Contains:** paper metadata, author data, citation links, institution data
- **Status:** VALUABLE, ACTIVE
- **Description:** **ACTIVE RESEARCH - PUBLICATION-FOCUSED** - Extensive work       on citation patterns, paper acceptance, and bibliometric analysis. The       308 subdirectories in cited_by_5_citing_5 suggest comprehensive citation       network analysis. High-quality visualizations with confidence intervals       indicate publication-ready analysis. Recent activity (October 2025).       Focus on computer science publications via OpenAlex. The award paper       analysis suggests research on scientific excellence
- **Related terms:** MAG replacement, academic literature, bibliography, citations, open access, papers, publications, research papers, scientific papers
- **Last Modified:** 2026-02-05

### hachi
- **Path:** `/project/jevans/hachi`
- **Owner:** hachicui94
- **Size:** 166 GB
- **Type:** Individual Researcher Directory - Extensive Data
- **Status:** VALUABLE
- **Description:** **VALUABLE DATA** - Large-scale bibliometric analysis focusing       on physics (APS) and computational linguistics (ACL). Contains substantial       author career trajectory data and field classifications. The CO2 project       suggests environmental/climate science analysis. Well-organized with clear       project separation. Multiple data formats (pickle, JSON, zip) suggest       diverse data processing pipelines. The career length and field mapping       files are particularly valuable for s
- **Last Modified:** 2024-07-14

### jamshid
- **Path:** `/project/jevans/jamshid`
- **Owner:** jsourati
- **Size:** 1-5 GB
- **Type:** Individual Researcher Directory - Computational Discovery
- **Status:** VALUABLE, ACTIVE
- **Description:** **ACTIVE RESEARCH - METHODOLOGICAL FOCUS** - Extensive work       on computational discovery methods, particularly abductive and       compositional reasoning. The multiple embedding projects (standard,       hyperbolic, materials-specific) suggest deep expertise in representation       learning. The Leverhulme project indicates major funded research. Work       spans multiple domains (biology, materials science, culture) with a       unified methodological approach. Recent activity (2024-2025).
- **Last Modified:** 2025-11-27

### junsol
- **Also known as:** OA, OpenAlex
- **Path:** `/project/jevans/junsol`
- **Owner:** junsol
- **Size:** 13 GB
- **Category:** Bibliographic Database
- **Type:** Individual Researcher Directory - LLM Benchmarking
- **Contains:** citation links, model configs, institution data, model weights, paper metadata, author data
- **Status:** VALUABLE, ACTIVE
- **Description:** **ACTIVE RESEARCH** - Comprehensive LLM benchmarking across       multiple cutting-edge models (DeepSeek R1, Llama 3.1, Qwen, QwQ). Very       recent work (March 2025) with latest models. The benchmark files contain       extensive evaluation results. The OpenAlex IGO (international governmental       organizations) data suggests research on how LLMs handle institutional       knowledge. Community notes work connects to misinformation and content       moderation. Valuable for understanding LLM
- **Related terms:** MAG replacement, ML models, academic literature, bibliography, citations, embeddings, machine learning, model weights, neural networks, open access, papers, pretrained models, publications, research papers, scientific papers
- **Last Modified:** 2025-10-01

### ningzili
- **Path:** `/project/jevans/ningzili`
- **Owner:** ningzi
- **Size:** 174 GB
- **Type:** Individual Researcher Directory - Ambiguity Analysis
- **Status:** VALUABLE, ACTIVE
- **Description:** **VALUABLE DATA** - Large-scale ambiguity analysis dataset       spanning multiple years with systematic partitioning. The 174GB size and       16K subdirectories suggest comprehensive temporal coverage. Ambiguity       analysis is valuable for understanding semantic uncertainty in text,       scientific language evolution, and concept clarity. Well-organized with       year-based structure and numbered partitions for parallel processing.       Recent activity (November-December 2024). High valu
- **Last Modified:** 2026-01-13

### Abenezer
- **Path:** `/project/jevans/Abenezer`
- **Owner:** abenezer
- **Size:** 167 MB
- **Type:** Individual Researcher Directory
- **Status:** VALUABLE
- **Description:** **VALUABLE DATA** - Curated dataset of NBER working papers       with complete bibliographic information. The parquet files suggest this       has been processed for analysis. The scraping code is functional but       typical research-grade code.
- **Last Modified:** 2025-04-30

### arxiv
- **Also known as:** arxiv, arXiv, biorxiv, medRxiv, ArXiv, medrxiv, BioRxiv, MedRxiv, bioRxiv
- **Path:** `/project/jevans/arxiv`
- **Owner:** jsourati
- **Size:** 4 GB
- **Category:** Preprint Server Data
- **Type:** Data Snapshot Directory
- **Contains:** abstracts, preprint metadata, categories
- **Status:** VALUABLE
- **Description:** **VALUABLE DATA** - Clean ArXiv metadata snapshot from 2023       plus bioRxiv and medRxiv preprint data. Well-documented data collection       process. Straightforward utility code. This is reference data that could       be useful for future projects needing preprint metadata, though data is       from 2023 and may need updating.
- **Related terms:** CS, arxiv papers, biology preprints, biomedical preprints, clinical preprints, computer science preprints, health sciences, life sciences, math, medical preprints, open access papers, physics, pre-prints, preprints, scientific preprints
- **Last Modified:** 2025-04-30

### carinak
- **Path:** `/project/jevans/carinak`
- **Owner:** carinakane
- **Size:** 50-100 MB
- **Type:** Individual Researcher Directory
- **Status:** VALUABLE, ACTIVE
- **Description:** Multiple active research projects focused on NLP and labor       economics ("Word Entropy", job market analysis). The Word Entropy project       appears substantial based on directory size. Recent activity through 2025.       Projects seem well-organized by topic. Would need to explore deeper to       assess data value, but the project organization suggests active, organized       research work.
- **Last Modified:** 2025-06-02

### cassietang
- **Path:** `/project/jevans/cassietang`
- **Owner:** cassietang
- **Size:** 3.5 GB
- **Type:** Individual Researcher Directory
- **Status:** VALUABLE
- **Description:** **VALUABLE DATA AND CODE** - Contains substantial NLP       resources including pre-trained Word2Vec embeddings and large-scale       cleaned/validated datasets. Multiple sentiment analysis implementations       with well-organized code. The "atypical combination" work suggests       research on novelty or creativity in text. Clean code structure with       clear file organization. The Klabpaper project appears to be a major       research effort with significant data processing. Useful for futu
- **Last Modified:** 2025-10-27

### jerryluo8
- **Path:** `/project/jevans/jerryluo8`
- **Owner:** jerryluo8
- **Size:** 100-500 MB
- **Type:** Individual Researcher Directory - Embeddings Focus
- **Status:** VALUABLE, ACTIVE
- **Description:** **ACTIVE RESEARCH** - Focused on creating embeddings for       multiple scientific domains (papers, patents, medical concepts). The       massive patent project (38 subdirs) appears to be the main focus. Recent       activity (2025). The combination of embeddings with topological methods       (persistent homology) and hypergraph analysis suggests sophisticated       mathematical approaches. Valuable for embedding methodology across domains,       particularly the patent embeddings given the sca
- **Last Modified:** 2025-11-07

### donghyun
- **Also known as:** dblp, DBLP, Dimensions
- **Path:** `/project/jevans/donghyun`
- **Owner:** kangd
- **Size:** 1-5 GB
- **Category:** Computer Science Bibliography
- **Type:** Individual Researcher Directory - Extensive Projects
- **Contains:** grant data, patent links, paper metadata, author data, venue data, citation data
- **Status:** VALUABLE, ACTIVE
- **Description:** **ACTIVE RESEARCH** - Extensive work on bibliometric analysis       with focus on concept hierarchies, conference analysis, and systematic       reviews. Multiple major projects including Cochrane medical systematic       reviews and computer science conference analysis. Well-organized with       clear project separation. The Assembly_Tree and Concept_Tree work suggests       hierarchical knowledge representation research. Has own DBLP snapshot and       Dimensions exploration, indicating seriou
- **Related terms:** CS papers, author collaboration, bibliography, citations, clinical trials, commercial database, comprehensive bibliometrics, computer science, conferences, grants, journals, papers, patents, policy documents, publications
- **Last Modified:** 2026-03-10

### 40123
- **Path:** `/project/jevans/40123`
- **Owner:** tzhang3
- **Size:** 11 GB
- **Type:** Educational/Tutorial Resources
- **Description:** Educational resources for natural language processing. Not       primary research data but useful reference materials for NLP projects.       Code is tutorial-level, not production research code.
- **Last Modified:** 2025-07-21

### perspective2025
- **Path:** `/project/jevans/perspective2025`
- **Owner:** panrui
- **Size:** 10-50 MB
- **Type:** Individual Researcher Directory
- **Status:** ACTIVE
- **Description:** Recent project (December 2025) on patent and Google Books       analysis with embedding methods. The bootstrap embedding note suggests       methodological development. Small scale, appears to be early-stage       research. Moderate value pending further development.
- **Last Modified:** 2026-03-10

### cnowak
- **Path:** `/project/jevans/cnowak`
- **Owner:** cnowak
- **Size:** 20-50 MB
- **Type:** Individual Researcher Directory
- **Description:** Appears to be a focused project on minority perspectives       research. Would need deeper exploration to assess content value. The       project name suggests DEI or social science research.
- **Last Modified:** 2025-06-25

### eduede
- **Path:** `/project/jevans/eduede`
- **Owner:** eduede
- **Size:** 10-20 MB
- **Type:** Individual Researcher Directory
- **Status:** ACTIVE
- **Description:** Small research directory from 2023-2025. The "abduction"       project suggests logic or reasoning research. Uses symlink to share data       from the tip project, indicating collaboration. Minimal scale, appears to       be focused exploratory work. Limited reuse value given small scope and       age.
- **Last Modified:** 2025-03-03

### callindai
- **Path:** `/project/jevans/callindai`
- **Owner:** callindai
- **Size:** <5 MB
- **Type:** Individual Researcher Directory
- **Description:** Minimal personal directory, likely just a workspace link to       the larger apto_data_engineering project. No independent value.
- **Last Modified:** 2025-06-13

### avi
- **Path:** `/project/jevans/avi`
- **Owner:** aoberoi1
- **Size:** 5-10 MB
- **Category:** ML Models
- **Type:** Individual Researcher Directory
- **Contains:** model configs, model weights
- **Status:** ACTIVE
- **Description:** Appears to be a course project on LLM fine-tuning using       Gemma2 models. Well-structured with proper cluster configuration files       and clean documentation (ACCELERATE_INTEGRATION.md, CLUSTER_SETUP_README.md).       Code looks organized and purposeful. Recent activity (December 2025). Not       particularly valuable for future reuse as it's course-specific work, but       demonstrates good software engineering practices.
- **Related terms:** ML models, embeddings, machine learning, model weights, neural networks, pretrained models, transformers
- **Last Modified:** 2026-02-23

### jlockhart
- **Path:** `/project/jevans/jlockhart`
- **Owner:** jlockhart
- **Size:** Unknown
- **Type:** Individual Researcher Directory (Restricted)
- **Description:** Directory exists but access is restricted. Cannot assess       contents or value. Note: This researcher also owns/co-owns the       gender_surprise directory (entry 026).
- **Last Modified:** 2026-01-26

### hongbofang
- **Path:** `/project/jevans/hongbofang`
- **Owner:** hongbofang
- **Size:** 200-500 MB
- **Category:** ML Models
- **Type:** Individual Researcher Directory
- **Contains:** model configs, model weights
- **Status:** ACTIVE
- **Description:** LLM and NLP research directory with focus on Chinese social       media (Weibo) and COVID-19 content. Multiple transformer model       implementations. The AI_index project suggests bibliometric work. MySQL       database indicates structured data storage. Mix of model development and       data analysis. Recent activity through 2025. Moderate value for NLP       methodology and COVID-19 social media research.
- **Related terms:** ML models, embeddings, machine learning, model weights, neural networks, pretrained models, transformers
- **Last Modified:** 2026-03-10

### beichen
- **Path:** `/project/jevans/beichen`
- **Owner:** beichenlu
- **Size:** 50-100 MB
- **Type:** Individual Researcher Directory
- **Status:** ACTIVE
- **Description:** Active ongoing research directory with very recent activity       (January 2026). Organized by date which suggests iterative development.       Contains substantial Jupyter notebooks indicating data analysis work. The       inequality project suggests economics or social science research. No       major datasets visible, primarily code and analysis. Utility depends on       what the specific research project is investigating.
- **Last Modified:** 2026-02-26

### renli
- **Path:** `/project/jevans/renli`
- **Owner:** renly
- **Size:** Unknown
- **Type:** Individual Researcher Directory (Restricted)
- **Description:** Restricted access directory. Cannot assess value.
- **Last Modified:** 2026-02-27

### hkling
- **Path:** `/project/jevans/hkling`
- **Owner:** hkling
- **Size:** 0 bytes
- **Type:** Empty Directory
- **Description:** Empty directory from 2022. No value.
- **Last Modified:** 2022-10-07

### qixin
- **Path:** `/project/jevans/qixin`
- **Owner:** qxlin
- **Size:** 100-500 MB
- **Category:** ML Models
- **Type:** Individual Researcher Directory
- **Contains:** model configs, model weights
- **Status:** ACTIVE
- **Description:** Master's thesis work on the Concordia framework (multi-agent       simulation) with LLM deployment using Ollama. Recent activity (February       2025). The multiple ollama service scripts show effort to run models       across different compute resources. Moderate research value, primarily       thesis-focused work on agent-based modeling.
- **Related terms:** ML models, embeddings, machine learning, model weights, neural networks, pretrained models, transformers
- **Last Modified:** 2025-02-20

### fengli
- **Also known as:** Microsoft Academic, MAG, Microsoft Academic Graph
- **Path:** `/project/jevans/fengli`
- **Owner:** fenglixu
- **Size:** 10-50 MB
- **Category:** Bibliographic Database (Historical)
- **Type:** Individual Researcher Directory
- **Contains:** paper metadata, author data, citation links
- **Description:** Small directory from 2022, appears to be legacy or historical       work related to MAG data analysis. Connection to haizi suggests this may       be archived work from a previous researcher. Limited activity, mainly       historical reference value.
- **Related terms:** academic literature, bibliography, citations, discontinued 2022, papers, publications, research papers, scientific papers
- **Last Modified:** 2025-04-30

### bready
- **Path:** `/project/jevans/bready`
- **Owner:** giochoi
- **Size:** <10 MB
- **Type:** Shared Project Directory
- **Description:** Small project directory focused on analyzing paper generation       patterns in AI research, tracking author positions. Very specific analysis       with limited scope. Data files suggest collaboration between gio and       hongbo. Limited reuse value outside this specific research question.
- **Last Modified:** 2026-03-04

### nadav_zhen
- **Path:** `/project/jevans/nadav_zhen`
- **Owner:** zhenzhang
- **Size:** 100-500 MB
- **Type:** Shared Project Directory
- **Status:** ACTIVE
- **Description:** Collaborative project between nadav and zhen on classification       and embedding tasks. Recent activity (June 2025). Well-organized pipeline       structure. Appears to be research on prompt engineering and question       generation. Moderate value for NLP methodology.
- **Last Modified:** 2025-06-06

### Other Researcher Directories

These directories exist but have no detailed catalog information:

- `00_cataloguer` (owner: akozlo, path: `/project/jevans/00_cataloguer`)
- `Honglin_Bao_share` (owner: honglinbao, path: `/project/jevans/Honglin_Bao_share`)
- `Volumes` (owner: akozlo, path: `/project/jevans/Volumes`)
- `YingrongMao` (owner: yingrong, path: `/project/jevans/YingrongMao`)
- `Yuxuan_Cai` (owner: yuxuanc, path: `/project/jevans/Yuxuan_Cai`)
- `arxiv_interp_graph` (owner: akozlo, path: `/project/jevans/arxiv_interp_graph`)
- `hongbo_jerry` (owner: hongbofang, path: `/project/jevans/hongbo_jerry`)
- `hongbo_share` (owner: hongbofang, path: `/project/jevans/hongbo_share`)
- `hongbo_zhen` (owner: zhenzhang, path: `/project/jevans/hongbo_zhen`)
- `jburchard` (owner: jburchard, path: `/project/jevans/jburchard`)
- `jmilbauer` (owner: 1608369274, path: `/project/jevans/jmilbauer`)
- `jojiao` (owner: jialingjiao, path: `/project/jevans/jojiao`)
- `joyyang` (owner: joyyang, path: `/project/jevans/joyyang`)
- `junsol_legacy` (owner: junsol, path: `/project/jevans/junsol_legacy`)
- `kgarden_backup` (owner: maxzhuyt, path: `/project/jevans/kgarden_backup`)
- `know_space` (owner: lyuzj, path: `/project/jevans/know_space`)
- `lilydong` (owner: lilydong, path: `/project/jevans/lilydong`)
- `linchen` (owner: linchen65, path: `/project/jevans/linchen`)
- `logs` (owner: shiyanglai, path: `/project/jevans/logs`)
- `matrix_files` (owner: hongkai, path: `/project/jevans/matrix_files`)
- `muhua` (owner: muhua, path: `/project/jevans/muhua`)
- `pca_results` (owner: avitalmintz, path: `/project/jevans/pca_results`)
- `pdfs` (owner: abenezer, path: `/project/jevans/pdfs`)
- `pdinesh` (owner: pdinesh, path: `/project/jevans/pdinesh`)
- `persona_vecs_TL` (owner: avitalmintz, path: `/project/jevans/persona_vecs_TL`)
- `ploertscher` (owner: akozlo, path: `/project/jevans/ploertscher`)
- `project` (owner: akozlo, path: `/project/jevans/project`)
- `projects-jerry` (owner: jerryluo8, path: `/project/jevans/projects-jerry`)
- `renli_shared` (owner: kangd, path: `/project/jevans/renli_shared`)
- `restricted_scores` (owner: jlockhart, path: `/project/jevans/restricted_scores`)
- `robbie` (owner: rbward, path: `/project/jevans/robbie`)
- `s2ag_hongbo` (owner: hongbofang, path: `/project/jevans/s2ag_hongbo`)
- `s2ag_ml_tables_text.parquet` (owner: danqingchen, path: `/project/jevans/s2ag_ml_tables_text.parquet`)
- `shared_with_jerry` (owner: jerryluo8, path: `/project/jevans/shared_with_jerry`)
- `sherryding` (owner: xding2, path: `/project/jevans/sherryding`)
- `sinanparmar` (owner: sinanparmar, path: `/project/jevans/sinanparmar`)
- `siyangwu` (owner: siyangwu, path: `/project/jevans/siyangwu`)
- `sumin` (owner: suminpark, path: `/project/jevans/sumin`)
- `team_rp` (owner: panrui, path: `/project/jevans/team_rp`)
- `tejaswini` (owner: tejaswini, path: `/project/jevans/tejaswini`)
- `trajectory-project` (owner: jerryluo8, path: `/project/jevans/trajectory-project`)
- `txts` (owner: abenezer, path: `/project/jevans/txts`)
- `tzhang3` (owner: tzhang3, path: `/project/jevans/tzhang3`)
- `untitled folder` (owner: nadavkunievsky, path: `/project/jevans/untitled folder`)
- `vrushank` (owner: blaiszik, path: `/project/jevans/vrushank`)
- `work` (owner: shiyanglai, path: `/project/jevans/work`)
- `xding2` (owner: xding2, path: `/project/jevans/xding2`)
- `xiaoliu` (owner: liuxiao, path: `/project/jevans/xiaoliu`)
- `yangyu` (owner: wangyd, path: `/project/jevans/yangyu`)
- `yanjingli` (owner: yanjingli, path: `/project/jevans/yanjingli`)
- `yuanyi` (owner: yuanyi, path: `/project/jevans/yuanyi`)
- `yurujiang` (owner: hongbofang, path: `/project/jevans/yurujiang`)
- `zhen_share` (owner: zhenzhang, path: `/project/jevans/zhen_share`)
- `zoeyfeng` (owner: zoeyfeng, path: `/project/jevans/zoeyfeng`)
- `zwang13` (owner: zwang13, path: `/project/jevans/zwang13`)
- `zzqian307` (owner: zzqian307, path: `/project/jevans/zzqian307`)

## Model Directories

Pre-trained models and model caches for machine learning.

### hf-cache
- **Path:** `/project/jevans/hf-cache`
- **Owner:** jdwoon0523
- **Size:** 20-50 GB
- **Category:** ML Models
- **Type:** Shared Resource Directory
- **Contains:** model configs, model weights
- **Status:** ACTIVE
- **Description:** Shared cache for Hugging Face transformer models. Contains       large LLMs (Llama 3.1 70B and 8B) and smaller models (BART, DistilBERT).       This saves download time and bandwidth for multiple users. Recent activity       (November 2025). Infrastructure directory rather than research data.       Valuable for avoiding redundant downloads but models can be re-downloaded       if needed.
- **Related terms:** ML models, embeddings, machine learning, model weights, neural networks, pretrained models, transformers
- **Last Modified:** 2025-11-09

### ollama_models
- **Path:** `/project/jevans/ollama_models`
- **Owner:** qxlin
- **Size:** 10-50 MB
- **Category:** ML Models
- **Type:** Shared Resource Directory
- **Contains:** model configs, model weights
- **Status:** ACTIVE
- **Description:** Ollama model cache directory for running LLMs locally. Contains       model blobs and manifests for the Ollama system. Recent use (April 2025).       Infrastructure directory - models can be re-downloaded. The world-writable       permissions allow shared access.
- **Related terms:** ML models, embeddings, machine learning, model weights, neural networks, pretrained models, transformers
- **Last Modified:** 2024-11-19

### models
- **Path:** `/project/jevans/models`
- **Owner:** maxzhuyt
- **Size:** 500 MB
- **Category:** ML Models
- **Type:** Shared Resource Directory
- **Contains:** model configs, model weights
- **Status:** ACTIVE
- **Description:** Repository for SPECTER2, a state-of-the-art citation-based       document embedding model from Allen AI. SPECTER2 creates embeddings that       capture semantic similarity based on citation patterns, making it valuable       for literature search, paper recommendation, and bibliometric analysis.       Multiple formats (ZIP archives, Hugging Face format, adapters) provide       flexibility. Recent setup (November 2024). Useful resource for anyone       working with scientific document embeddings.
- **Related terms:** ML models, embeddings, machine learning, model weights, neural networks, pretrained models, transformers
- **Last Modified:** 2025-11-14

### LLMs
- **Path:** `/project/jevans/LLMs`
- **Owner:** akozlo
- **Size:** 100-200 GB
- **Category:** ML Models
- **Type:** Shared Resource Directory
- **Contains:** model configs, model weights
- **Status:** ACTIVE
- **Description:** Shared lab cache for large language models. Contains multiple       versions of Gemma (1B to 27B) and Llama 3.1 (8B and 70B). The quantized       ChatQA model enables running large models on limited hardware. Recent       activity (February-April 2025). Infrastructure directory that saves       significant download time/bandwidth for multiple researchers. Models can       be re-downloaded if needed but having local copies is convenient. The       variety of model sizes enables experimentation ac
- **Related terms:** ML models, embeddings, machine learning, model weights, neural networks, pretrained models, transformers
- **Last Modified:** 2026-01-16

