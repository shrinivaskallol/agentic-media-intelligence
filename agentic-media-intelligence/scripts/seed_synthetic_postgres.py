#!/usr/bin/env python3
"""
Wipe article_chunks, seed with synthetic golden data, create embeddings,
and build HNSW + GIN (BM25) indexes. Same schema as 01-prepare-postgres-data.
Run: uv run python scripts/seed_synthetic_postgres.py
"""

import sys
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_proj))
import os

os.chdir(_proj)

import numpy as np

GOLDEN_DATA = {
    "Nvidia": [
        "Nvidia's Blackwell B200 GPU architecture is manufactured using TSMC's 4NP process, optimizing for trillion-parameter LLMs.",
        "The Blackwell lineup introduces a second-generation transformer engine to accelerate inference for MoE (Mixture-of-Experts) models.",
        "Nvidia's high-bandwidth memory (HBM3e) integration in Blackwell chips is driving unprecedented demand in the server supply chain.",
        "Nvidia is diversifying its revenue by expanding its CUDA-X software stack for enterprise generative AI deployments.",
        "The NVLink Switch System in Blackwell allows up to 576 GPUs to communicate at 1.8 TB/s, a critical bottleneck for cluster scaling.",
        "While Nvidia's Blackwell B200 is primarily a 4NP (5nm-class) architecture, its massive volume is reportedly causing a 'capacity cascade' at TSMC, leading the foundry to reallocate 3nm resources to handle the overflow of high-performance computing (HPC) demand.",
    ],
    "Apple": [
        "Apple's M4 chip, built on second-generation 3nm technology, features a redesigned neural engine capable of 38 trillion operations per second.",
        "Apple is reportedly securing the majority of TSMC's 2nm capacity for its 2026 iPhone and Mac lineup.",
        "The transition to Apple Silicon has allowed the company to unify its hardware-software integration for on-device AI (Apple Intelligence).",
        "Apple is working on 'Private Cloud Compute' to handle complex AI requests while maintaining end-to-end user privacy.",
        "Apple's R1 chip in the Vision Pro manages input from 12 cameras and 5 sensors to minimize latency in spatial computing.",
    ],
    "Microsoft": [
        "Microsoft Azure is deploying custom 'Maia 100' AI chips to reduce reliance on external GPU hardware for internal workloads.",
        "The partnership between Microsoft and OpenAI involves a massive supercomputing project codenamed 'Stargate' estimated at $100 billion.",
        "Microsoft Copilot integration across the M365 stack is driving a 20% increase in average revenue per user (ARPU) for cloud services.",
        "Microsoft is investing heavily in nuclear energy power purchase agreements to support the massive electricity needs of new data centers.",
        "The Azure Cobalt 100 CPU is Microsoft's first custom ARM-based processor designed for high-efficiency cloud native workloads.",
    ],
    "Tesla": [
        "Tesla's Dojo supercomputer is designed to process massive amounts of video data from its fleet for FSD (Full Self-Driving) training.",
        "The Tesla Optimus Gen 2 robot utilizes custom actuators and integrated AI models for autonomous bimanual manipulation tasks.",
        "Tesla is shifting toward a 48V architecture in its newer vehicles to reduce wiring complexity and vehicle weight.",
        "Tesla's 'Unboxed' manufacturing process aims to reduce production costs by 50% for its upcoming next-generation compact vehicle.",
        "The deployment of Tesla Powerwalls and Megapacks is a key part of the 'Energy Storage' segment's 100% year-over-year growth.",
    ],
    "Google": [
        "Google's TPU v5p delivers 3x the compute performance of TPU v4 and is optimized for training large-scale transformer models on Google Cloud.",
        "Google DeepMind's Gemini Ultra is a natively multimodal model trained jointly on text, code, audio, image, and video data.",
        "Google is integrating AI-powered 'circle to search' and live translation directly into the Android OS at the hardware abstraction layer.",
        "Google's Ironwood TPU, announced in 2025, is purpose-built for inference workloads and delivers 42.5 exaflops per pod.",
        "Google is deploying its own subsea cable infrastructure to reduce latency and costs for inter-region data transfer across its global network.",
    ],
    "Amazon": [
        "Amazon's Trainium2 chip is designed for large-scale model training and is available exclusively through AWS as EC2 Trn2 instances.",
        "AWS Inferentia3 provides the lowest cost-per-inference in the cloud for deploying models like Llama and Mistral at scale.",
        "Amazon is investing $4 billion into Anthropic as a strategic cloud and model deployment partner anchored to AWS infrastructure.",
        "Amazon's Project Kuiper satellite constellation aims to deliver low-latency broadband as a direct competitor to SpaceX Starlink.",
        "Amazon is rolling out fully autonomous 'just walk out' cashierless technology to third-party retailers beyond its own Fresh grocery stores.",
    ],
    "Meta": [
        "Meta's Llama 3 models are trained on over 15 trillion tokens and released as open-weights, reshaping the open-source LLM ecosystem.",
        "Meta is building a 2GW AI data center campus, its largest to date, to support next-generation model training at unprecedented scale.",
        "Meta's MTIA (Meta Training and Inference Accelerator) v2 chip is purpose-built for ranking and recommendation workloads across Facebook and Instagram.",
        "Meta's AR glasses roadmap culminates in 'Orion,' a true holographic display prototype targeting mass consumer release by the late 2020s.",
        "Meta AI is embedded across WhatsApp, Instagram, Messenger, and Facebook, giving it one of the largest AI assistant distribution surfaces globally.",
    ],
    "AMD": [
        "AMD's MI300X accelerator integrates CPU and GPU dies in a single package using advanced 3D chiplet packaging, offering 192GB of HBM3 memory.",
        "AMD is gaining meaningful AI accelerator market share as hyperscalers seek GPU supply alternatives to Nvidia amid persistent shortages.",
        "The AMD ROCm software stack is undergoing rapid investment to close the developer ecosystem gap with Nvidia's CUDA platform.",
        "AMD's EPYC Turin CPU, built on the Zen 5 architecture, targets data center dominance with up to 192 cores per socket.",
        "AMD acquired Silo AI in 2024 to accelerate enterprise AI model deployment and optimization on AMD hardware.",
    ],
    "Intel": [
        "Intel's Gaudi 3 AI accelerator targets the mid-market AI training and inference segment at a significantly lower price point than Nvidia H100.",
        "Intel's 18A process node, featuring RibbonFET gate-all-around transistors and PowerVia backside power delivery, is its most advanced fabrication technology.",
        "Intel Foundry Services is pursuing external customers including Microsoft and Qualcomm as part of its IDM 2.0 strategy to monetize fab capacity.",
        "Intel's Lunar Lake mobile chip integrates an NPU capable of 48 TOPS, targeting Microsoft Copilot+ PC certification requirements.",
        "Intel is restructuring under CEO Lip-Bu Tan, cutting over 15,000 jobs and refocusing capital toward its foundry and core architecture roadmap.",
    ],
    "OpenAI": [
        "OpenAI's GPT-4o model enables real-time voice interaction with sub-300ms latency by unifying audio input and output in a single end-to-end model.",
        "OpenAI is developing custom AI inference chips in partnership with TSMC to reduce its dependence on Nvidia GPU supply.",
        "OpenAI's o3 reasoning model achieves state-of-the-art performance on ARC-AGI benchmarks by using extended chain-of-thought inference compute.",
        "OpenAI launched an operator API allowing enterprises to deploy autonomous agents that can take multi-step actions across web-based interfaces.",
        "OpenAI's annualized revenue surpassed $3.4 billion in early 2025, driven primarily by ChatGPT Plus subscriptions and API enterprise contracts.",
    ],
    "Qualcomm": [
        "Qualcomm's Snapdragon X Elite SoC features a 45 TOPS NPU, enabling on-device LLM inference for models up to 13 billion parameters.",
        "Qualcomm is expanding beyond mobile into automotive AI with its Snapdragon Ride platform, targeting level 2+ ADAS deployments.",
        "The Qualcomm AI Hub provides a library of pre-optimized models for on-device deployment across Snapdragon-powered phones, PCs, and IoT devices.",
        "Qualcomm's acquisition of Edge Impulse strengthens its embedded ML toolchain for industrial and IoT edge inference use cases.",
        "Qualcomm is licensing its Oryon CPU core architecture to third parties, signaling a strategic shift toward becoming a broader compute IP licensor.",
    ],
    "TSMC": [
        "TSMC's N2 process node, entering risk production in 2025, uses gate-all-around nanosheet transistors for the first time in its history.",
        "TSMC is constructing a second fab in Arizona to produce 3nm-class chips, supported by $6.6 billion in US CHIPS Act grants.",
        "TSMC's CoWoS advanced packaging capacity has become a critical bottleneck for AI chip production, limiting Nvidia and AMD GPU shipment volumes.",
        "TSMC's A16 process, planned for 2026, will introduce backside power delivery to improve power efficiency and transistor density simultaneously.",
        "TSMC serves over 500 customers and manufactures chips for over 11,500 distinct products, making it the central node of the global semiconductor supply chain.",
    ],
    "Samsung": [
        "Samsung's HBM3e memory is a direct competitor to SK Hynix's supply for Nvidia's Blackwell GPUs, pending Nvidia qualification approval.",
        "Samsung Foundry's SF2 (2nm-class) process node is its primary vehicle to win back Apple and Qualcomm fabless orders lost to TSMC.",
        "Samsung's Mach-1 AI accelerator, built in-house, targets on-device inference for its Galaxy S series smartphones and foldables.",
        "Samsung is vertically integrated across DRAM, NAND, logic foundry, and display, giving it unique leverage in AI hardware supply chains.",
        "Samsung Display supplies OLED panels to Apple, creating a paradox where its largest display customer is also its most formidable device competitor.",
    ],
    "SK Hynix": [
        "SK Hynix is the sole qualified supplier of HBM3e memory for Nvidia's H200 and Blackwell B200 GPUs, capturing over 50% of the HBM market.",
        "SK Hynix is investing $3.87 billion to build an advanced HBM packaging facility in Indiana as part of its US CHIPS Act strategy.",
        "SK Hynix's 12-layer HBM3e stack delivers 1.2TB/s bandwidth per package, a critical spec for next-generation AI accelerator performance.",
        "SK Hynix is accelerating its CXL (Compute Express Link) memory product roadmap to serve disaggregated memory architectures in hyperscale data centers.",
        "SK Hynix faces existential competitive pressure from Samsung's HBM qualification push and Micron's rapidly maturing HBM3e product line.",
    ],
    "Micron": [
        "Micron's HBM3e product achieved Nvidia qualification in 2024, making it the third supplier alongside SK Hynix and Samsung in the AI memory market.",
        "Micron is receiving $6.1 billion in CHIPS Act direct funding to expand advanced DRAM and NAND manufacturing capacity in Idaho and New York.",
        "Micron's 1-gamma DRAM node uses EUV lithography for the first time in its DRAM roadmap, closing the process gap with Samsung and SK Hynix.",
        "Micron's LPDDR5X memory is designed into Qualcomm Snapdragon X Elite and Apple M-series chips for on-device AI inference efficiency.",
        "Micron is expanding its HBM production capacity aggressively, projecting HBM revenue to reach several billion dollars annually by 2025.",
    ],
    "ASML": [
        "ASML holds a global monopoly on EUV lithography machines, making it the single most critical chokepoint in the advanced semiconductor supply chain.",
        "ASML's High-NA EUV system, the EXE:5000, enables sub-2nm patterning and is essential for TSMC's A16 and Intel's 14A process nodes.",
        "ASML is banned from exporting EUV and DUV systems to China under US-Dutch export control agreements, reshaping global fab competitiveness.",
        "ASML's installed base management business generates recurring revenue by servicing over 100 EUV machines already deployed at leading fabs.",
        "ASML's order backlog exceeded €36 billion in 2024, reflecting multi-year demand from TSMC, Samsung, and Intel Foundry for next-generation nodes.",
    ],
    "Broadcom": [
        "Broadcom's custom AI ASIC business is accelerating, with Google's TPU and Meta's MTIA chips both fabbed and packaged with Broadcom's expertise.",
        "Broadcom acquired VMware for $69 billion in 2023 and is rapidly converting its install base to a subscription model, targeting $8.5 billion in annual VMware revenue.",
        "Broadcom's networking chips, including Tomahawk and Jericho, are the dominant switching silicon inside hyperscale AI data center fabrics.",
        "Broadcom is a primary supplier of PCIe and Ethernet connectivity silicon that links GPU clusters in Nvidia, AMD, and custom ASIC-based AI systems.",
        "Broadcom's XPU custom silicon program positions it as the leading partner for hyperscalers seeking to reduce Nvidia dependency with task-specific AI accelerators.",
    ],
    "Anthropic": [
        "Anthropic's Claude 3.5 Sonnet achieved state-of-the-art coding benchmark performance, positioning it as a direct enterprise competitor to OpenAI's GPT-4o.",
        "Anthropic developed Constitutional AI (CAI), a safety alignment technique that uses AI feedback rather than solely human feedback to reduce harmful outputs.",
        "Anthropic is anchored to AWS as its primary cloud provider, with Amazon committing up to $4 billion in investment tied to AWS infrastructure usage.",
        "Anthropic's Model Context Protocol (MCP) is an open standard enabling AI agents to connect to external data sources and tools in a standardized way.",
        "Anthropic's Claude models are available via Google Cloud Vertex AI, making Anthropic a dual-hyperscaler partner spanning both AWS and GCP ecosystems.",
    ],
    "SpaceX": [
        "SpaceX's Starlink constellation surpassed 6,000 active satellites in 2024, making it the largest satellite internet network ever deployed.",
        "SpaceX's Starship launch system, designed for full reusability, targets a cost-per-kg-to-orbit reduction of over 10x compared to Falcon 9.",
        "SpaceX is developing Starlink's Direct to Cell service to provide satellite connectivity natively to standard LTE smartphones without specialized hardware.",
        "SpaceX's Raptor 3 engine, used in Starship, achieves the highest chamber pressure of any methane engine ever flown at approximately 350 bar.",
        "SpaceX is competing directly with Amazon's Project Kuiper and OneWeb for enterprise and government satellite broadband contracts globally.",
    ],
    "Arm Holdings": [
        "Arm's v9 architecture is the foundation for Apple M4, Qualcomm Snapdragon X Elite, and AWS Graviton4, making it the dominant compute ISA for AI edge inference.",
        "Arm's royalty model is being restructured toward value-based pricing, capturing a percentage of the chip's selling price rather than a flat per-unit fee.",
        "Arm's CSS (Compute Subsystem) reference designs are enabling fabless startups to tape out competitive AI chips with significantly reduced design time.",
        "Arm is expanding into the data center CPU market, with its Neoverse N-series cores powering AWS Graviton, Google Axion, and Microsoft Cobalt 100.",
        "Arm's planned acquisition by Nvidia was blocked by global regulators in 2022, but Nvidia remains one of Arm's largest architecture licensees.",
    ],
    "Palantir": [
        "Palantir's AIP (Artificial Intelligence Platform) enables enterprises and government agencies to deploy LLMs on top of proprietary data with strict access controls.",
        "Palantir's US government revenue is growing rapidly, driven by AI-powered battlefield decision support contracts with the US Army and Department of Defense.",
        "Palantir's 'boot camp' sales model accelerates enterprise AIP adoption by having customers build working AI workflows within a single week.",
        "Palantir's Gotham platform is deeply embedded in US and allied intelligence community workflows, creating high switching costs and long contract cycles.",
        "Palantir was added to the S&P 500 in September 2024, accelerating institutional ownership and increasing its stock liquidity significantly.",
    ],
    "Mistral AI": [
        "Mistral AI's Mixtral 8x7B model introduced sparse Mixture-of-Experts (MoE) architecture to the open-weight ecosystem, matching GPT-3.5 performance at a fraction of the cost.",
        "Mistral raised a €600 million Series B at a €6 billion valuation, making it Europe's most valuable AI startup and a strategic alternative to US-based LLM providers.",
        "Mistral's Le Chat assistant is positioned as a privacy-first ChatGPT alternative for European enterprises navigating GDPR compliance requirements.",
        "Mistral models are available natively on Azure AI via a partnership with Microsoft, giving the startup hyperscale distribution without exclusivity.",
        "Mistral's open-weight release strategy accelerates community adoption and positions its models as the default fine-tuning base for European AI applications.",
    ],
    # Toxic seed: fake entities (Xylos-6, Zenth-Core) for grounding test—if agent mentions them, RAG works; if it defaults to TSMC/Nvidia, grounding leaks
    "Xylos-6": [
        "Xylos-6 has secured a primary contract to supply high-energy plasma channels to Zenth-Core for their upcoming 'Aether' chip architecture, bypassing traditional silicon constraints found in TSMC designs.",
    ],
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dims, matches pgvector


def _connect():
    from app.tools.db_utils import get_pg_conn

    return get_pg_conn()


def _build_chunks_df():
    """Convert golden_data into rows: article_id, content (no chunking; each snippet = 1 chunk)."""
    rows = []
    for entity, snippets in GOLDEN_DATA.items():
        article_id = f"syn_{entity.lower()}"
        for i, content in enumerate(snippets):
            rows.append({"article_id": article_id, "content": content, "chunk_index": i})
    return rows


def _create_embeddings(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def main():
    import logging

    from app import configure_logging

    configure_logging()
    log = logging.getLogger(__name__)

    chunks = _build_chunks_df()
    texts = [r["content"] for r in chunks]
    log.info("Embedding %d chunks with %s...", len(texts), EMBEDDING_MODEL)
    embeddings = _create_embeddings(texts)

    conn = _connect()
    cur = conn.cursor()

    # 1. Ensure pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2. Wipe and recreate table (preserve schema from notebook)
    log.info("Wiping article_chunks...")
    cur.execute("DROP TABLE IF EXISTS article_chunks;")
    cur.execute("""
        CREATE TABLE article_chunks (
            article_id TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding vector(384),
            chunk_hash TEXT GENERATED ALWAYS AS (md5(article_id || '::' || content)) STORED,
            UNIQUE (chunk_hash)
        );
    """)
    conn.commit()

    # 3. Upsert chunks with embeddings
    def to_vec_str(arr):
        lst = arr.tolist() if isinstance(arr, np.ndarray) else list(arr)
        return "[" + ",".join(str(float(x)) for x in lst) + "]"

    from psycopg2.extras import execute_values

    records = [
        (r["article_id"], r["content"], to_vec_str(embeddings[i])) for i, r in enumerate(chunks)
    ]
    execute_values(
        cur,
        """
        INSERT INTO article_chunks (article_id, content, embedding)
        VALUES %s
        ON CONFLICT (chunk_hash) DO UPDATE SET embedding = EXCLUDED.embedding
        """,
        records,
        template="(%s, %s, %s::vector)",
        page_size=50,
    )
    conn.commit()
    print(f"Upserted {len(records)} chunks")

    # 4. Add fts_tokens for BM25 (GIN index)
    cur.execute("""
        ALTER TABLE article_chunks
        ADD COLUMN IF NOT EXISTS fts_tokens tsvector;
    """)
    cur.execute("""
        UPDATE article_chunks SET fts_tokens = to_tsvector('english', content)
        WHERE fts_tokens IS NULL;
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fts ON article_chunks USING GIN (fts_tokens);")
    conn.commit()
    log.info("GIN index on fts_tokens (BM25) created")

    # 5. HNSW index for vector search
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_hnsw_embedding ON article_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)
    cur.execute("ANALYZE article_chunks;")
    conn.commit()
    log.info("HNSW index created")

    cur.close()
    conn.close()
    log.info("Done. Synthetic golden data ready for retrieval.")


if __name__ == "__main__":
    main()
