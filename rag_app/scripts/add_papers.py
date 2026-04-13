#!/usr/bin/env python3
"""Add new papers to metadata.csv for corpus expansion.

Usage:
    python scripts/add_papers.py              # Dry-run: show what would be added
    python scripts/add_papers.py --apply      # Actually append to metadata.csv

After running with --apply, rebuild the index:
    cd vendor/KohakuRAG
    kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
"""

import csv
import sys
from pathlib import Path

METADATA_CSV = Path(__file__).resolve().parent.parent / "data" / "metadata.csv"

# ============================================================================
# NEW PAPERS TO ADD
# Format: (id, type, title, year, citation, url)
# ============================================================================

NEW_PAPERS = [
    # --- Water usage / footprint ---
    (
        "ren2024",
        "paper",
        "Reconciling the Contrasting Narratives on the Environmental Impact of Large Language Models",
        "2024",
        "Shaolei Ren, Bill Tomlinson, Rebecca W. Black, A. Torrance (2024). Reconciling the contrasting narratives on the environmental impact of large language models. Scientific Reports. https://arxiv.org/pdf/2409.07116",
        "https://arxiv.org/pdf/2409.07116",
    ),
    (
        "li2024_water",
        "paper",
        "Towards Sustainable GenAI using Generation Directives for Carbon-Friendly Large Language Model Inference",
        "2024",
        "Baolin Li, Yankai Jiang, Vijay Gadepally, Devesh Tiwari (2024). Towards Sustainable GenAI using Generation Directives for Carbon-Friendly Large Language Model Inference. arXiv. https://arxiv.org/pdf/2403.12900",
        "https://arxiv.org/pdf/2403.12900",
    ),
    # --- Datacenter sustainability ---
    (
        "acun2023",
        "paper",
        "Carbon Explorer: A Holistic Framework for Designing Carbon Aware Datacenters",
        "2023",
        "Bilge Acun, Benjamin Lee, Fiodar Kazhamiaka, Kiwan Maeng, Udit Gupta, Manoj Chakkaravarthy, David Brooks, Carole-Jean Wu (2023). Carbon Explorer: A Holistic Framework for Designing Carbon Aware Datacenters. ASPLOS '23. https://arxiv.org/pdf/2210.02681",
        "https://arxiv.org/pdf/2210.02681",
    ),
    (
        "radovanovic2022",
        "paper",
        "Carbon-Aware Computing for Datacenters",
        "2022",
        "Ana Radovanovic, Ross Koningstein, Ian Schneider, Bokan Chen, Alexandre Duber, Binz Roy, David Talaber, Drew Ferguson, Nic Tills, Kathy Zhu, Max Nova, Jared Chen, Ken Hua (2022). Carbon-Aware Computing for Datacenters. IEEE TPDS. https://arxiv.org/pdf/2106.11750",
        "https://arxiv.org/pdf/2106.11750",
    ),
    # --- GPU / hardware energy efficiency ---
    (
        "desislavov2023",
        "paper",
        "Trends in AI Inference Energy Consumption: Beyond the Performance-vs-Parameter Laws of Deep Learning",
        "2023",
        "Radosvet Desislavov, Fernando Martinez-Plumed, Jose Hernandez-Orallo (2023). Trends in AI Inference Energy Consumption: Beyond the Performance-vs-Parameter Laws of Deep Learning. Sustainable Computing. https://arxiv.org/pdf/2301.00774",
        "https://arxiv.org/pdf/2301.00774",
    ),
    (
        "samsi2023_gpu",
        "paper",
        "Benchmarking Large Language Models on Supercomputers",
        "2023",
        "Siddharth Samsi, Dan Zhao, Andrew Gittens, David Bader, Vijay Gadepally (2023). Benchmarking Large Language Models on Supercomputers. arXiv. https://arxiv.org/pdf/2402.05065",
        "https://arxiv.org/pdf/2402.05065",
    ),
    # --- Inference optimization for sustainability ---
    (
        "xu2024",
        "paper",
        "A Survey on Model Compression for Large Language Models",
        "2024",
        "Xunyu Zhu, Jian Li, Yong Liu, Can Ma, Weiping Wang (2024). A Survey on Model Compression for Large Language Models. TACL. https://arxiv.org/pdf/2308.07633",
        "https://arxiv.org/pdf/2308.07633",
    ),
    (
        "stojkovic2024",
        "paper",
        "Towards Greener LLMs: Bringing Energy-Efficiency to the Forefront of LLM Inference",
        "2024",
        "Jovan Stojkovic, Esha Choukse, Chaojie Zhang, Inigo Goiri, Josep Torrellas (2024). Towards Greener LLMs: Bringing Energy-Efficiency to the Forefront of LLM Inference. arXiv. https://arxiv.org/pdf/2403.20306",
        "https://arxiv.org/pdf/2403.20306",
    ),
    (
        "chavan2024",
        "paper",
        "Faster and Lighter LLMs: A Survey on Current Challenges and Way Forward",
        "2024",
        "Arnav Chavan, Raghav Magazine, Shubham Kushwaha, Mérouane Debbah, Deepak Gupta (2024). Faster and Lighter LLMs: A Survey on Current Challenges and Way Forward. IJCAI. https://arxiv.org/pdf/2402.01799",
        "https://arxiv.org/pdf/2402.01799",
    ),
    # --- Carbon footprint / lifecycle analysis ---
    (
        "gupta2022",
        "paper",
        "Chasing Carbon: The Elusive Environmental Footprint of Computing",
        "2022",
        "Udit Gupta, Young Geun Kim, Sylvia Lee, Jordan Tse, Hsien-Hsin S. Lee, Gu-Yeon Wei, David Brooks, Carole-Jean Wu (2022). Chasing Carbon: The Elusive Environmental Footprint of Computing. HPCA '22. https://arxiv.org/pdf/2011.02839",
        "https://arxiv.org/pdf/2011.02839",
    ),
    (
        "faiz2024",
        "paper",
        "LLMCarbon: Modeling the End-to-End Carbon Footprint of Large Language Models",
        "2024",
        "Ahmad Faiz, Sotaro Kaneda, Ruhan Wang, Rita Osi, Prateek Sharma, Fan Chen, Lei Jiang (2024). LLMCarbon: Modeling the End-to-End Carbon Footprint of Large Language Models. ICLR '24. https://arxiv.org/pdf/2309.14393",
        "https://arxiv.org/pdf/2309.14393",
    ),
    (
        "lannelongue2021",
        "paper",
        "Green Algorithms: Quantifying the Carbon Footprint of Computation",
        "2021",
        "Loic Lannelongue, Jason Grealey, Michael Inouye (2021). Green Algorithms: Quantifying the Carbon Footprint of Computation. Advanced Science. https://arxiv.org/pdf/2007.07610",
        "https://arxiv.org/pdf/2007.07610",
    ),
    # --- Recent 2025-2026 papers ---
    (
        "bommasani2024",
        "paper",
        "The Foundation Model Transparency Index v1.1",
        "2024",
        "Rishi Bommasani, Kevin Klyman, Shayne Longpre, Betty Xiong, Sayash Kapoor, Nestor Maslej, Arvind Narayanan, Percy Liang (2024). The Foundation Model Transparency Index v1.1. arXiv. https://arxiv.org/pdf/2407.12929",
        "https://arxiv.org/pdf/2407.12929",
    ),
    (
        "bannour2024",
        "paper",
        "A Systematic Review of Green AI",
        "2024",
        "Nada Bannour, Sahar Ghannay, Aurelien Nedelec, Anne Vilnat (2024). A Systematic Review of Green AI. Artificial Intelligence Review. https://arxiv.org/pdf/2301.11047",
        "https://arxiv.org/pdf/2301.11047",
    ),
    (
        "tomlinson2024",
        "paper",
        "The Carbon Emissions of Writing and Illustrating Are Lower for AI than for Humans",
        "2024",
        "Bill Tomlinson, Rebecca W. Black, Donald J. Patterson, Andrew W. Torrance (2024). The Carbon Emissions of Writing and Illustrating Are Lower for AI than for Humans. Scientific Reports. https://arxiv.org/pdf/2303.06219",
        "https://arxiv.org/pdf/2303.06219",
    ),
    # --- Additional water-focused papers ---
    (
        "george2023",
        "paper",
        "Measuring the Environmental Impacts of Artificial Intelligence Compute and Applications",
        "2023",
        "Sasha Luccioni, Alex Hernandez-Garcia, Jesse Dodge (2023). Measuring the Environmental Impacts of Artificial Intelligence Compute and Applications. arXiv. https://arxiv.org/pdf/2211.02001",
        "https://arxiv.org/pdf/2211.02001",
    ),
    # =====================================================================
    # BATCH 2 — Papers found via Semantic Scholar / arXiv search (2024-2026)
    # =====================================================================
    # --- Water usage ---
    (
        "shumba2024",
        "paper",
        "A Water Efficiency Dataset for African Data Centers",
        "2024",
        "Noah Shumba, Opelo Tshekiso, Pengfei Li, Giulia Fanti, Shaolei Ren (2024). A Water Efficiency Dataset for African Data Centers. arXiv. https://arxiv.org/pdf/2412.03716",
        "https://arxiv.org/pdf/2412.03716",
    ),
    # --- Datacenter sustainability ---
    (
        "islam2025",
        "paper",
        "Electricity Demand and Grid Impacts of AI Data Centers: Challenges and Prospects",
        "2025",
        "Mohammad A. Islam et al. (2025). Electricity Demand and Grid Impacts of AI Data Centers: Challenges and Prospects. arXiv. https://arxiv.org/pdf/2509.07218",
        "https://arxiv.org/pdf/2509.07218",
    ),
    (
        "waste2energy2025",
        "paper",
        "Waste-to-Energy-Coupled AI Data Centers: Cooling Efficiency and Grid Resilience",
        "2025",
        "(2025). Waste-to-Energy-Coupled AI Data Centers: Cooling Efficiency and Grid Resilience. arXiv. https://arxiv.org/pdf/2512.24683",
        "https://arxiv.org/pdf/2512.24683",
    ),
    (
        "carbonrt2024",
        "paper",
        "Carbon Footprint Reduction for Sustainable Data Centers in Real-Time",
        "2024",
        "(2024). Carbon Footprint Reduction for Sustainable Data Centers in Real-Time. arXiv. https://arxiv.org/pdf/2403.14092",
        "https://arxiv.org/pdf/2403.14092",
    ),
    (
        "dccooling2025",
        "paper",
        "Data Center Cooling System Optimization Using Offline Reinforcement Learning",
        "2025",
        "(2025). Data Center Cooling System Optimization Using Offline Reinforcement Learning. arXiv. https://arxiv.org/pdf/2501.15085",
        "https://arxiv.org/pdf/2501.15085",
    ),
    # --- GPU / hardware efficiency ---
    (
        "h100power2025",
        "paper",
        "Empirically-Calibrated H100 Node Power Models",
        "2025",
        "(2025). Empirically-Calibrated H100 Node Power Models. arXiv. https://arxiv.org/pdf/2506.14551",
        "https://arxiv.org/pdf/2506.14551",
    ),
    (
        "hwswcodesign2025",
        "paper",
        "Sustainable AI Training via Hardware-Software Co-Design",
        "2025",
        "(2025). Sustainable AI Training via Hardware-Software Co-Design. arXiv. https://arxiv.org/pdf/2508.13163",
        "https://arxiv.org/pdf/2508.13163",
    ),
    (
        "intelperwatt2025",
        "paper",
        "Intelligence Per Watt: Measuring Intelligence Efficiency of Local AI",
        "2025",
        "(2025). Intelligence Per Watt: Measuring Intelligence Efficiency of Local AI. arXiv. https://arxiv.org/pdf/2511.07885",
        "https://arxiv.org/pdf/2511.07885",
    ),
    (
        "promptspower2025",
        "paper",
        "From Prompts to Power: Measuring the Energy Footprint of LLM Inference",
        "2025",
        "(2025). From Prompts to Power: Measuring the Energy Footprint of LLM Inference. arXiv. https://arxiv.org/pdf/2511.05597",
        "https://arxiv.org/pdf/2511.05597",
    ),
    # --- Inference optimization ---
    (
        "quantbatch2026",
        "paper",
        "Understanding Efficiency: Quantization, Batching, and Serving Strategies in LLM Energy Use",
        "2026",
        "(2026). Understanding Efficiency: Quantization, Batching, and Serving Strategies in LLM Energy Use. arXiv. https://arxiv.org/pdf/2601.22362",
        "https://arxiv.org/pdf/2601.22362",
    ),
    (
        "edgequant2025",
        "paper",
        "Sustainable LLM Inference for Edge AI: Evaluating Quantized LLMs for Energy Efficiency",
        "2025",
        "(2025). Sustainable LLM Inference for Edge AI: Evaluating Quantized LLMs for Energy Efficiency, Output Accuracy, and Inference Latency. arXiv. https://arxiv.org/pdf/2504.03360",
        "https://arxiv.org/pdf/2504.03360",
    ),
    (
        "dvfs2025",
        "paper",
        "Investigating Energy Efficiency and Performance Trade-offs in LLM Inference Across Tasks and DVFS Settings",
        "2025",
        "(2025). Investigating Energy Efficiency and Performance Trade-offs in LLM Inference Across Tasks and DVFS Settings. arXiv. https://arxiv.org/pdf/2501.08219",
        "https://arxiv.org/pdf/2501.08219",
    ),
    # --- Carbon lifecycle analysis ---
    (
        "elsworth2025",
        "paper",
        "Life-Cycle Emissions of AI Hardware: A Cradle-To-Grave Approach and Generational Trends",
        "2025",
        "Cooper Elsworth et al. (2025). Life-Cycle Emissions of AI Hardware: A Cradle-To-Grave Approach and Generational Trends. arXiv. https://arxiv.org/pdf/2502.01671",
        "https://arxiv.org/pdf/2502.01671",
    ),
    (
        "falk2025",
        "paper",
        "More than Carbon: Cradle-to-Grave Environmental Impacts of GenAI Training on the Nvidia A100 GPU",
        "2025",
        "Sophia Falk et al. (2025). More than Carbon: Cradle-to-Grave Environmental Impacts of GenAI Training on the Nvidia A100 GPU. arXiv. https://arxiv.org/pdf/2509.00093",
        "https://arxiv.org/pdf/2509.00093",
    ),
    (
        "beyondeff2024",
        "paper",
        "Beyond Efficiency: Scaling AI Sustainably",
        "2024",
        "(2024). Beyond Efficiency: Scaling AI Sustainably. arXiv. https://arxiv.org/pdf/2406.05303",
        "https://arxiv.org/pdf/2406.05303",
    ),
    (
        "scopingreview2025",
        "paper",
        "Toward Sustainable Generative AI: A Scoping Review of Carbon Footprint",
        "2025",
        "(2025). Toward Sustainable Generative AI: A Scoping Review of Carbon Footprint. arXiv. https://arxiv.org/pdf/2511.17179",
        "https://arxiv.org/pdf/2511.17179",
    ),
    # --- Recent 2025-2026 comprehensive ---
    (
        "aiservers2026",
        "paper",
        "The Environmental Impact of AI Servers and Sustainable Solutions",
        "2026",
        "(2026). The Environmental Impact of AI Servers and Sustainable Solutions. arXiv. https://arxiv.org/pdf/2601.06063",
        "https://arxiv.org/pdf/2601.06063",
    ),
    (
        "google2025",
        "paper",
        "Measuring the Environmental Impact of Delivering AI at Google Scale",
        "2025",
        "(2025). Measuring the Environmental Impact of Delivering AI at Google Scale. arXiv. https://arxiv.org/pdf/2508.15734",
        "https://arxiv.org/pdf/2508.15734",
    ),
    (
        "greenaisurvey2025",
        "paper",
        "Green AI: A Systematic Review and Meta-Analysis of Its Definitions, Lifecycle Models, Hardware and Measurement Attempts",
        "2025",
        "(2025). Green AI: A Systematic Review and Meta-Analysis. arXiv. https://arxiv.org/pdf/2511.07090",
        "https://arxiv.org/pdf/2511.07090",
    ),
    # =====================================================================
    # BATCH 3 — Cloud vs on-prem, water, and author-network papers
    # =====================================================================
    # --- Cloud vs on-premise / infrastructure ---
    (
        "treviso2022",
        "paper",
        "The Cost of Training NLP Models: A Concise Overview",
        "2022",
        "Marcos Treviso, Ji-Ung Lee, Tianchu Ji, Betty van Aken, Qingqing Cao, Manuel R. Ciosici, Michael Hassid, Kenneth Heafield, Sara Hooker, Colin Raffel, Pedro H. Martins, Andre F. T. Martins, Jessica Zosa Forde, Peter Grabowski, Chester Palen-Michel, Ann Clifton, Yufang Hou, Hal Daume III, Iryna Gurevych, Roy Schwartz (2022). The Cost of Training NLP Models: A Concise Overview. arXiv. https://arxiv.org/pdf/2206.07160",
        "https://arxiv.org/pdf/2206.07160",
    ),
    (
        "skypilot2023",
        "paper",
        "SkyPilot: An Intercloud Broker for Sky Computing",
        "2023",
        "Zongheng Yang, Zhanghao Wu, Michael Luo, Wei-Lin Chiang, Romil Bhardwaj, Woosuk Kwon, Siyuan Zhuang, Frank Sifei Luan, Gautam Mittal, Scott Shenker, Ion Stoica (2023). SkyPilot: An Intercloud Broker for Sky Computing. NSDI '23. https://arxiv.org/pdf/2205.07147",
        "https://arxiv.org/pdf/2205.07147",
    ),
    (
        "spotserve2024",
        "paper",
        "SpotServe: Serving Generative Large Language Models on Preemptible Instances",
        "2024",
        "Xupeng Miao, Chunan Shi, Jiangfei Duan, Xiaoli Xi, Dahua Lin, Bin Cui, Zhihao Jia (2024). SpotServe: Serving Generative Large Language Models on Preemptible Instances. ASPLOS '24. https://arxiv.org/pdf/2311.15566",
        "https://arxiv.org/pdf/2311.15566",
    ),
    (
        "sia2023",
        "paper",
        "Sia: Heterogeneity-aware, Goodput-optimized ML-Cluster Scheduling",
        "2023",
        "Suhas Jayaram Subramanya, Daiyaan Arfeen, Shouxu Lin, Aurick Qiao, Zhihao Jia, Gregory R. Ganger (2023). Sia: Heterogeneity-aware, Goodput-optimized ML-Cluster Scheduling. SOSP '23. https://arxiv.org/pdf/2306.00954",
        "https://arxiv.org/pdf/2306.00954",
    ),
    (
        "pollux2022",
        "paper",
        "Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning",
        "2022",
        "Aurick Qiao, Sang Keun Choe, Suhas Jayaram Subramanya, Willie Neiswanger, Qirong Ho, Hao Zhang, Gregory R. Ganger, Eric P. Xing (2022). Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning. OSDI '21. https://arxiv.org/pdf/2008.12260",
        "https://arxiv.org/pdf/2008.12260",
    ),
    # --- Water usage / cooling / water reduction ---
    (
        "waterscarf2025",
        "paper",
        "Not All Water Consumption Is Equal: A Water Stress Weighted Metric for Sustainable Computing",
        "2025",
        "(2025). Not All Water Consumption Is Equal: A Water Stress Weighted Metric for Sustainable Computing. arXiv. https://arxiv.org/pdf/2506.22773",
        "https://arxiv.org/pdf/2506.22773",
    ),
    (
        "islami2024",
        "paper",
        "Environmental Impact and Policy Implications of AI-Driven Water Consumption in Data Centers",
        "2024",
        "Md Amimul Ihsan Aquil, Md Arafat Alam, Ahmed Imteaj (2024). Environmental Impact and Policy Implications of AI-Driven Water Consumption in Data Centers. arXiv. https://arxiv.org/pdf/2412.13853",
        "https://arxiv.org/pdf/2412.13853",
    ),
    (
        "lcopt2025",
        "paper",
        "LC-Opt: Benchmarking RL and Agentic AI for End-to-End Liquid Cooling Optimization in Data Centers",
        "2025",
        "(2025). LC-Opt: Benchmarking Reinforcement Learning and Agentic AI for End-to-End Liquid Cooling Optimization in Data Centers. arXiv. https://arxiv.org/pdf/2511.00116",
        "https://arxiv.org/pdf/2511.00116",
    ),
    (
        "greenllm2025",
        "paper",
        "Green-LLM: Optimal Workload Allocation for Environmentally-Aware Distributed Inference",
        "2025",
        "Jiaming Cheng, D. Nguyen (2025). Green-LLM: Optimal Workload Allocation for Environmentally-Aware Distributed Inference. arXiv. https://arxiv.org/pdf/2505.02309",
        "https://arxiv.org/pdf/2505.02309",
    ),
    (
        "moore2025",
        "paper",
        "Sustainable Carbon-Aware and Water-Efficient LLM Scheduling in Geo-Distributed Cloud Datacenters",
        "2025",
        "Hayden Moore, Sirui Qi, Ninad Hogade, D. Milojicic, Cullen E. Bash, S. Pasricha (2025). Sustainable Carbon-Aware and Water-Efficient LLM Scheduling in Geo-Distributed Cloud Datacenters. GLSVLSI '25. https://arxiv.org/pdf/2501.14334",
        "https://arxiv.org/pdf/2501.14334",
    ),
    # --- Papers citing key corpus authors (Luccioni, Strubell, Ren, Wu) ---
    (
        "luccioni2022bloom",
        "paper",
        "Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model",
        "2023",
        "Alexandra Sasha Luccioni, Sylvain Viguier, Anne-Laure Ligozat (2023). Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model. JMLR. https://arxiv.org/pdf/2211.02001",
        "https://arxiv.org/pdf/2211.02001",
    ),
    (
        "ligozat2022",
        "paper",
        "Unraveling the Hidden Environmental Impacts of AI Solutions for Environment",
        "2022",
        "Anne-Laure Ligozat, Julien Lefevre, Aurelie Bugeau, Jacques Combaz (2022). Unraveling the Hidden Environmental Impacts of AI Solutions for Environment Life Cycle Assessment of AI Solutions. Sustainability. https://arxiv.org/pdf/2110.11822",
        "https://arxiv.org/pdf/2110.11822",
    ),
    (
        "henderson2020",
        "paper",
        "Towards the Systematic Reporting of the Energy and Carbon Footprints of Machine Learning",
        "2020",
        "Peter Henderson, Jieru Hu, Joshua Romoff, Emma Brunskill, Dan Jurafsky, Joelle Pineau (2020). Towards the Systematic Reporting of the Energy and Carbon Footprints of Machine Learning. JMLR. https://arxiv.org/pdf/2002.05651",
        "https://arxiv.org/pdf/2002.05651",
    ),
    (
        "lacoste2019",
        "paper",
        "Quantifying the Carbon Emissions of Machine Learning",
        "2019",
        "Alexandre Lacoste, Alexandra Luccioni, Victor Schmidt, Thomas Dandres (2019). Quantifying the Carbon Emissions of Machine Learning. arXiv. https://arxiv.org/pdf/1910.09700",
        "https://arxiv.org/pdf/1910.09700",
    ),
]


def load_existing_ids() -> set[str]:
    """Load existing document IDs from metadata.csv."""
    ids = set()
    if METADATA_CSV.exists():
        with open(METADATA_CSV, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                ids.add(row["id"].strip())
    return ids


def main():
    apply = "--apply" in sys.argv
    existing = load_existing_ids()

    # Deduplicate: skip papers already in corpus, and skip duplicate IDs/URLs
    to_add = []
    seen_ids = set()
    seen_urls = set()
    skipped = []

    for paper in NEW_PAPERS:
        pid, ptype, title, year, citation, url = paper
        if pid in existing:
            skipped.append((pid, "already in corpus"))
            continue
        if pid in seen_ids:
            skipped.append((pid, "duplicate ID in this batch"))
            continue
        if url in seen_urls:
            skipped.append((pid, f"duplicate URL: {url}"))
            continue
        seen_ids.add(pid)
        seen_urls.add(url)
        to_add.append(paper)

    print(f"\nExisting papers: {len(existing)}")
    print(f"New papers to add: {len(to_add)}")
    if skipped:
        print(f"Skipped: {len(skipped)}")
        for pid, reason in skipped:
            print(f"  - {pid}: {reason}")

    print()
    for pid, ptype, title, year, citation, url in to_add:
        print(f"  [{year}] {pid}: {title[:70]}...")

    if not to_add:
        print("\nNothing to add.")
        return

    if not apply:
        print(f"\nDry run — pass --apply to actually write {len(to_add)} papers to {METADATA_CSV}")
        return

    # Append to CSV
    with open(METADATA_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for paper in to_add:
            writer.writerow(paper)

    print(f"\nAppended {len(to_add)} papers to {METADATA_CSV}")
    print(f"Total papers: {len(existing) + len(to_add)}")
    print(
        "\nNext steps:\n"
        "  1. cd vendor/KohakuRAG\n"
        "  2. kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py\n"
        "  3. Copy the new wattbot_jinav4.db to your PPVC\n"
    )


if __name__ == "__main__":
    main()
