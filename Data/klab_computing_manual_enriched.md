# UChicago Knowledge Lab Computing Manual (Enriched Edition)

**Version:** Enriched Edition with Appendices
**Last Updated:** January 2026
**Original Document:** KLab Computing Manual (Living Document)

---

## How to Use This Document

This enriched manual extends the original KLab Computing Manual with detailed appendices containing verified information from official sources. Each section includes citations to primary sources that Chorus can reference when answering questions.

**For quick reference:** Use the main sections (same as original manual)
**For detailed technical information:** Consult the relevant appendix
**For citations:** Each appendix includes source URLs and access dates

---

## Table of Contents

- [Compute Resources](#compute-resources)
  - [Midway3](#midway3)
  - [Knowledge Garden](#knowledge-garden)
  - [K-Garden 1080](#k-garden-1080)
  - [SSCS Acropolis](#sscs-acropolis)
- [Data Sets](#data-sets)
- [Midway Software and Application Tips](#midway-software-and-application-tips)
- [Appendices](#appendices)
  - [Appendix A: Midway3 Complete Reference](#appendix-a-midway3-complete-reference)
  - [Appendix B: SLURM Command Reference](#appendix-b-slurm-command-reference)
  - [Appendix C: GPU Specifications and Selection Guide](#appendix-c-gpu-specifications-and-selection-guide)
  - [Appendix D: Storage and Quota Management](#appendix-d-storage-and-quota-management)
  - [Appendix E: Software Modules Reference](#appendix-e-software-modules-reference)
  - [Appendix F: Python and Jupyter Configuration](#appendix-f-python-and-jupyter-configuration)
  - [Appendix G: Data Sources Documentation](#appendix-g-data-sources-documentation)
  - [Appendix H: SSCS Acropolis Reference](#appendix-h-sscs-acropolis-reference)
  - [Appendix I: RCC Allocations Guide](#appendix-i-rcc-allocations-guide)

---

# Compute Resources

## Midway3

The University of Chicago's main research computing cluster managed by the Research Computing Center (RCC).

### Connection Information

**SSH Connection:**
```bash
ssh <CNetID>@midway3.rcc.uchicago.edu
```

**Authentication:** Two-factor authentication via Duo MFA is required per University security guidelines.

**Official Documentation:** https://docs.rcc.uchicago.edu/ssh/main

> **Source:** RCC User Guide - SSH Connection (https://docs.rcc.uchicago.edu/ssh/main), accessed January 2026

### SLURM Job Scheduler

Midway uses the SLURM (Simple Linux Utility for Resource Management) workload manager. Key principles:

- Request appropriate resources to share the cluster effectively
- For single-threaded code, request 1-2 CPUs
- Request only the GPUs your code will actually use
- Use `--exclusive` only when you need an entire node's resources

**See:** [Appendix B: SLURM Command Reference](#appendix-b-slurm-command-reference) for complete command documentation.

### Nodes

A node is a physical server/computer with specific capabilities:

| Node Type | Specifications | Use Case |
|-----------|---------------|----------|
| **Login nodes** | Internet access, limited compute | Downloading data, submitting jobs |
| **caslake** | 48 cores, 192 GB RAM | General computing |
| **bigmem** | 48 cores, 768 GB-2 TB RAM | Memory-intensive workloads |
| **amd** | 128 cores, 256 GB RAM (AMD EPYC) | Highly parallel workloads |
| **gpu** | Various GPUs (RTX6000, V100, A100, H100) | Deep learning, GPU computing |

**See:** [Appendix C: GPU Specifications and Selection Guide](#appendix-c-gpu-specifications-and-selection-guide) for detailed GPU comparisons.

### Partitions

#### KLab-Accessible Partitions

| Partition | Account | Nodes | Walltime | Notes |
|-----------|---------|-------|----------|-------|
| **caslake** | pi-jevans | 184 nodes (48 cores, 192 GB each) | 36 hours | General purpose |
| **gpu** | pi-jevans | 11 nodes with various GPUs | 36 hours | Use `--constraint` to specify GPU type |
| **bigmem** | pi-jevans | 2 nodes (768 GB and 1536 GB) | 36 hours | Memory-intensive work |
| **amd** | pi-jevans | 40 nodes (128 cores, 256 GB) | 36 hours | High core count |
| **ssd** | ssd | 18 compute nodes | 36 hours | Social Sciences Division |
| **ssd-gpu** | ssd | 1 node with 4x A100 | 36 hours | Set `--qos=ssd` |
| **jevans** | pi-jevans | 4 bigmem nodes (1.5-2 TB RAM) | 48 hours | Lab exclusive |
| **jevans-gpu** | pi-jevans | 1 node with 4x H100 (94 GB VRAM each) | 48 hours | Lab exclusive, GPU only |

> **Source:** RCC User Guide - Partitions (https://docs.rcc.uchicago.edu/partitions/), accessed January 2026

#### Interactive vs Batch Jobs

**Batch Jobs** (`sbatch`): Set-and-forget execution. Best for production runs.
```bash
sbatch ./job_script.sbatch
```

**Interactive Jobs** (`sinteractive`): Real-time interaction. Best for development and debugging.
```bash
sinteractive --account=pi-jevans --partition=caslake --time=02:00:00
```

**See:** [Appendix B: SLURM Command Reference](#appendix-b-slurm-command-reference) for complete examples.

### Storage Space

| Location | Quota | Backup | Purpose |
|----------|-------|--------|---------|
| `/home/$USER` | 30 GB soft / 35 GB hard | Yes | Personal files |
| `/project/jevans/` | 100 TB (shared) | Yes | Lab shared storage |
| `/scratch/midway3/$USER` | 5 TB | **No** | Temporary large files |

**Check your quota:**
```bash
quota
```

**See:** [Appendix D: Storage and Quota Management](#appendix-d-storage-and-quota-management) for detailed guidance.

> **Source:** RCC User Guide - Storage (https://docs.rcc.uchicago.edu/storage/main/), accessed January 2026

### Compute Allocation

The lab receives yearly allocations from RCC (renewed each September).

**Check allocation:**
```bash
accounts allocations  # View current allocation
accounts balance      # View remaining balance
```

**See:** [Appendix I: RCC Allocations Guide](#appendix-i-rcc-allocations-guide) for allocation types and renewal process.

> **Source:** RCC Accounts & Allocations (https://rcc.uchicago.edu/accounts-allocations/request-allocation), accessed January 2026

---

## Knowledge Garden

A bare metal server physically in the Knowledge Lab.

| Specification | Value |
|--------------|-------|
| GPU | 1x NVIDIA 3090 |
| CPU | 48 cores |
| RAM | 126 GB |
| Access | SSH to 205.208.1.121 (via VPN or campus network) |

**Getting access:** Contact someone with sudo powers (Austin, Jamshid, Donghyun, or others)

**Usage etiquette:** Monitor resources with `htop` and `nvitop`; close programs when finished.

---

## K-Garden 1080

A bare metal server physically in the Knowledge Lab.

| Specification | Value |
|--------------|-------|
| GPUs | 4x NVIDIA 1080 Ti |
| CPU | 12 cores |
| RAM | 126 GB |
| Access | SSH to 205.208.1.203 (via VPN or campus network) |

**Getting access:** Contact someone with sudo powers (Austin, Jeff, Donghyun, Junsol)

---

## SSCS Acropolis

A cluster specifically for social science research.

| Feature | Details |
|---------|---------|
| Access | SSH to acropolis.uchicago.edu |
| Job Scheduler | Moab/Torque (PBS), not SLURM |
| Head Node | Dell R920, 40 cores (80 threads), Intel Xeon E7-8891 v2 @ 3.20GHz |
| Resource Limits | Up to 350 cores and 1 TB memory per user |
| Storage | `/home`, `/share` (10GbE network), `/scratch` (local SSD per node) |

**Important:** Files on Acropolis are NOT visible on Midway and vice versa.

**See:** [Appendix H: SSCS Acropolis Reference](#appendix-h-sscs-acropolis-reference) for PBS commands.

> **Source:** SSCS Acropolis FAQ (https://sscs.uchicago.edu/category/faq/cluster), accessed January 2026

---

# Data Sets

## Bibliometric and Academic Data

### Microsoft Academic Graph (MAG)
- **Location:** `/project/jevans/MAG_Dec_2021_snapshot/`
- **Description:** December 2021 snapshot (Microsoft discontinued MAG in 2021)
- **Schema:** See included HTML documentation file

### OpenAlex
- **Location:** `/project/jevans/tip/data/openalex/` (October 2023 snapshot)
- **Raw data:** `/project/jevans/tip/data/openalex/data`
- **Format:** Parquet files available
- **Description:** Open catalog of 240M+ scholarly works, authors, institutions, and their connections. Successor to MAG.

**See:** [Appendix G: Data Sources Documentation](#appendix-g-data-sources-documentation) for API access details.

> **Source:** OpenAlex Documentation (https://docs.openalex.org), accessed January 2026

### Web of Science
- **Location:** `/project/jevans/tip/data/wos_2023/` (XML and Parquet formats)
- **Alternative:** SSCS Cronus system (contact may be outdated)
- **Note:** API access requires institutional license

> **Source:** Clarivate Developer Portal (https://developer.clarivate.com/apis/wos), accessed January 2026

## Research Administration Data

### IRIS UMETRICS
- **Access:** Requires Data Use Agreement with University of Michigan
- **Contact:** IRIS at University of Michigan to add names to DUA
- **Description:** Administrative data from 580,000+ funded awards worth $192 billion, payments to 1.2M+ vendors, wages to ~985,000 employees (2024 release)
- **Data scope:** University HR, sponsored projects, procurement data linked to publications, patents, dissertations
- **Access method:** Virtual Data Enclave (VDE) after IRB approval and training

**See:** [Appendix G: Data Sources Documentation](#appendix-g-data-sources-documentation) for detailed access procedures.

> **Source:** IRIS - Institute for Research on Innovation & Science (https://iris.isr.umich.edu/), accessed January 2026

## Social Media Data

### Reddit Dumps
- **Historical (onset to 2023-02):** `/project/jevans/hongkai/reddit` (collected by Hongkai Mao)
- **Recent (2024-04 to 2025-04):** `/project/jevans/reddit_data` (collected by Ruining He)

**Future maintenance:** Use aria2c to download from Academic Torrents (https://academictorrents.com/) on AWS EC2 instances, then transfer via scp.

> **Source:** Academic Torrents (https://academictorrents.com/), accessed January 2026

## Biomedical Data

### PubMed Knowledge Graph Version 2 (PKG)
- **Location:** `/project/jevans/PKG_v2_2023`
- **Description:** Comprehensive knowledge graph with 36M+ papers, 1.3M patents, 0.48M clinical trials
- **Linkages:** 482M biomedical entity links, 19M citation links, 7M project links
- **Use cases:** Knowledge reasoning, scientometrics, biomedical NLP

> **Source:** PubMed Knowledge Graph (https://pubmedkg.github.io/), Scientific Data publication, accessed January 2026

## Other Datasets
- **ProQuest TDM:** Newspapers, press releases (UChicago access)
- **Refinitiv:** Earnings call transcripts (UChicago access)
- **Peer Review Data:** `/project/jevans/Honglin_Bao_share/peer_review` (CS conference reviews with scores and confidence)

---

# Midway Software and Application Tips

## Helpful SLURM Commands

| Command | Description |
|---------|-------------|
| `myq` | Show your current running and pending jobs |
| `squeue \| grep jevans` | Show all jobs on jevans partitions |
| `sinfo \| grep jevans` | Show node status (idle, busy, mixed) |
| `scontrol show job [jobid]` | Detailed job information |

**See:** [Appendix B: SLURM Command Reference](#appendix-b-slurm-command-reference) for complete command list.

## Installing Software

**Important:** Do NOT install software in your home directory. It will quickly exceed your 30 GB quota.

**Best practice:**
1. Check if software is available as a module: `module avail [name]`
2. Load the module: `module load [name]`
3. Only install additional packages after loading the base module

**See:** [Appendix E: Software Modules Reference](#appendix-e-software-modules-reference) for module commands.

> **Source:** RCC User Guide - Software (https://docs.rcc.uchicago.edu/software/), accessed January 2026

## Jupyter Lab

**See:** [Appendix F: Python and Jupyter Configuration](#appendix-f-python-and-jupyter-configuration) for complete setup instructions.

**Quick start:**
```bash
# Start interactive session
sinteractive --account=pi-jevans --partition=caslake --time=04:00:00

# Get IP and start Jupyter
HOST_IP=$(/sbin/ip route get 8.8.8.8 | awk '{print $7;exit}')
module load python/anaconda-2022.05
jupyter lab --no-browser --port=18765 --ip=$HOST_IP
```

## PySpark

**See the original manual** for detailed PySpark setup on SLURM, including multi-node configurations.

## MySQL Database

**See the original manual** for complete MySQL setup instructions on Midway, including configuration files and Python connector usage.

---

# Appendices

---

## Appendix A: Midway3 Complete Reference

### SSH Connection Details

**Basic connection:**
```bash
ssh <CNetID>@midway3.rcc.uchicago.edu
```

**Authentication process:**
1. Enter password (no characters displayed while typing)
2. Complete Duo MFA verification

**First connection:** Accept the host key by typing `yes` when prompted.

**Important notes:**
- Login nodes have internet access but are NOT for computing
- Compute nodes have no internet access
- SSH key authentication available only to PIs with justified need (contact RCC helpdesk)

### Data Transfer

**SCP (Secure Copy):**
```bash
scp <localFile> <CNetID>@midway3.rcc.uchicago.edu:<remotePath>
scp <CNetID>@midway3.rcc.uchicago.edu:<remoteFile> <localPath>
```

**SFTP:** Recommended for interactive transfers; supported by Termius client

**Rsync:** Faster for synchronizing directories
```bash
rsync -avz <localDir>/ <CNetID>@midway3.rcc.uchicago.edu:<remoteDir>/
```

### Complete Partition Specifications

| Partition | Nodes | Cores/Node | Memory/Node | GPUs | Time Limit | QoS Limits |
|-----------|-------|------------|-------------|------|------------|------------|
| caslake | 184 | 48 (gold-6248r) | 192 GB | - | 36h | 100 nodes, 4800 CPUs max |
| gpu (A100) | 1 | varies | 384 GB | 4x A100 | 36h | 12 jobs max |
| gpu (V100) | 5 | varies | 192 GB | 4x V100 | 36h | 12 jobs max |
| gpu (RTX6000) | 5 | varies | 192 GB | 4x RTX6000 | 36h | 12 jobs max |
| bigmem | 2 | 48 | 768 GB / 1536 GB | - | 36h | - |
| amd | 40 | 128 (epyc-7702) | 256 GB | - | 36h | 64 nodes, 128 CPUs max |
| amd-hm | 1 | 128 | 2048 GB | - | 36h | - |
| ssd | 18 | varies | varies | - | 36h | Use account=ssd |
| ssd-gpu | 1 | varies | varies | 4x A100 | 36h | Use qos=ssd |
| jevans | 4 | 48 | 1.5-2 TB | - | 48h | Lab exclusive |
| jevans-gpu | 1 | 32 | 500 GB | 4x H100 (94 GB) | 48h | Lab exclusive |

> **Source:** RCC User Guide - Partitions (https://docs.rcc.uchicago.edu/partitions/), accessed January 2026

---

## Appendix B: SLURM Command Reference

### Job Submission (sbatch)

**Basic submission:**
```bash
sbatch ./job_script.sbatch
```

**Essential #SBATCH flags:**

| Flag | Purpose | Example |
|------|---------|---------|
| `--job-name` | Job identifier | `--job-name=myanalysis` |
| `--account` | Billing account | `--account=pi-jevans` |
| `--partition` | Target partition | `--partition=caslake` |
| `--time` | Maximum runtime | `--time=1-03:30:00` (d-hh:mm:ss) |
| `--nodes` | Number of nodes | `--nodes=2` |
| `--ntasks-per-node` | Tasks per node | `--ntasks-per-node=48` |
| `--cpus-per-task` | CPUs per task | `--cpus-per-task=4` |
| `--mem-per-cpu` | Memory per CPU (MB) | `--mem-per-cpu=4000` |
| `--mem` | Total memory | `--mem=64G` |
| `--gres` | Generic resources | `--gres=gpu:1` |
| `--constraint` | Node constraints | `--constraint=v100` |
| `--exclusive` | Exclusive node access | `--exclusive` |
| `--output` | stdout file | `--output=job_%j.out` |
| `--error` | stderr file | `--error=job_%j.err` |
| `--mail-type` | Email notifications | `--mail-type=END,FAIL` |
| `--mail-user` | Email address | `--mail-user=you@uchicago.edu` |
| `--array` | Job array | `--array=1-100` or `--array=1-100%10` |
| `--dependency` | Job dependencies | `--dependency=afterok:12345` |

### Job Templates

**Single-core CPU job:**
```bash
#!/bin/bash
#SBATCH --job-name=single-cpu
#SBATCH --account=pi-jevans
#SBATCH --partition=caslake
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000
#SBATCH --time=04:00:00

module load python/anaconda-2022.05
python my_script.py
```

**GPU job:**
```bash
#!/bin/bash
#SBATCH --job-name=gpu-job
#SBATCH --account=pi-jevans
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

module load python/anaconda-2022.05
module load cuda
python train_model.py
```

**Job array (parallel independent tasks):**
```bash
#!/bin/bash
#SBATCH --job-name=array-job
#SBATCH --account=pi-jevans
#SBATCH --partition=caslake
#SBATCH --array=1-100%20
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000
#SBATCH --time=01:00:00

# Access array task ID
TASK_ID=$SLURM_ARRAY_TASK_ID
python process_file.py --task-id $TASK_ID
```

### Interactive Jobs (sinteractive)

**Basic interactive session:**
```bash
sinteractive --account=pi-jevans --partition=caslake --time=02:00:00
```

**With specific resources:**
```bash
sinteractive --account=pi-jevans --partition=gpu --gres=gpu:1 --constraint=v100 --time=04:00:00
```

**BigMem node:**
```bash
sinteractive --account=pi-jevans --partition=bigmem --ntasks=1 --cpus-per-task=8 --mem=128G --time=04:00:00
```

**Debug QoS (15 minutes, no allocation charge):**
```bash
sinteractive --qos=debug --time=00:15:00 --ntasks=2 --account=pi-jevans
```

### Job Monitoring Commands

| Command | Description |
|---------|-------------|
| `squeue --user=$USER` | Your running/pending jobs |
| `squeue --partition=jevans` | Jobs on specific partition |
| `squeue --state=PENDING` | All pending jobs |
| `sinfo --partition=caslake` | Partition node status |
| `scontrol show job <jobid>` | Detailed job info |
| `sacct -j <jobid>` | Job accounting data |
| `scancel <jobid>` | Cancel a job |
| `scancel --user=$USER` | Cancel all your jobs |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `$SLURM_JOB_ID` | Unique job identifier |
| `$SLURM_ARRAY_TASK_ID` | Array task number |
| `$SLURM_CPUS_PER_TASK` | Allocated CPUs |
| `$SLURM_NUM_NODES` | Total allocated nodes |
| `$SLURM_NTASKS_PER_NODE` | Tasks per node |
| `$SLURM_SUBMIT_DIR` | Directory where job was submitted |
| `$TMPDIR` / `$SLURM_TMPDIR` | Local scratch directory |

> **Source:** RCC User Guide - sbatch (https://docs.rcc.uchicago.edu/slurm/sbatch/), sinteractive (https://docs.rcc.uchicago.edu/slurm/sinteractive/), accessed January 2026

---

## Appendix C: GPU Specifications and Selection Guide

### GPUs Available on Midway

| GPU Model | VRAM | Memory Bandwidth | Architecture | Tensor Cores | Best For |
|-----------|------|------------------|--------------|--------------|----------|
| **H100** | 80-94 GB HBM3 | 3.35 TB/s (SXM) | Hopper | 528 | Large LLMs, multi-GPU training |
| **A100** | 40-80 GB HBM2e | 2 TB/s | Ampere | 432 | Large models, mixed-precision |
| **V100** | 32 GB HBM2 | 900 GB/s | Volta | 640 | General deep learning |
| **RTX 6000** | 24 GB GDDR6 | 672 GB/s | Ampere | 336 | Smaller models, inference |
| **RTX 3090** | 24 GB GDDR6X | 936 GB/s | Ampere | 328 | Development, smaller training |
| **1080 Ti** | 11 GB GDDR5X | 484 GB/s | Pascal | - | Legacy, smaller models |

### GPU Selection Guidelines

**For Large Language Models (LLMs):**
- 70B+ parameters: Use H100 (jevans-gpu) or multiple A100s
- 7B-70B parameters: A100 preferred, V100 or 2-3x RTX 6000 acceptable
- <7B parameters: Any GPU sufficient

**For Training:**
- H100 is ~2-3x faster than A100 for transformer training
- A100 is ~1.5-2x faster than V100
- A100 SXM4 is ~58% faster than RTX A6000 for deep learning

**For Inference:**
- VRAM is primary constraint (must fit model)
- Memory bandwidth affects throughput
- Similar tokens/second: 1x H100 ≈ 2x A100 ≈ 3x RTX 6000

**Requesting specific GPUs:**
```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100  # Options: rtx6000, v100, a100
```

### GPU Monitoring

**Check GPU usage in real-time:**
```bash
nvitop  # Recommended: shows memory, utilization, processes
nvidia-smi  # Basic GPU status
watch -n 1 nvidia-smi  # Continuous monitoring
```

### Multi-GPU Considerations

- Many PyTorch scripts use only 1 GPU by default
- Multi-GPU may not provide speedup without proper parallelization
- Some LLM inference can use multiple GPUs with tensor parallelism
- Always verify your code actually uses multiple GPUs before requesting them

> **Sources:**
> - NVIDIA H100 Datasheet (https://www.nvidia.com/en-us/data-center/h100/)
> - GPU Benchmarks for Deep Learning (https://bizon-tech.com/gpu-benchmarks/)
> - Accessed January 2026

---

## Appendix D: Storage and Quota Management

### Storage Locations and Quotas

| Location | Soft Quota | Hard Quota | Backed Up | Purpose |
|----------|-----------|------------|-----------|---------|
| `/home/$USER` | 30 GB / 300K files | 35 GB / 1M files | Yes | Personal files, configs |
| `/project/jevans/` | 100 TB (shared) | - | Yes | Lab shared data |
| `/scratch/midway3/$USER` | 100 GB | 5 TB | **No** | Temporary large files |

### Checking Your Quota

```bash
quota                    # Quick summary
quota -u $USER           # Detailed user quota
rcchelp quota            # RCC-specific command
```

**Understanding quota output:**
- **blocks**: Storage space used
- **files**: Number of files (inodes)
- **soft limit**: Can exceed temporarily (grace period)
- **hard limit**: Cannot exceed; writes will fail

### When Over Quota

**Identify the problem:**
- Over on **files**: Compress/archive small files into fewer large files
- Over on **blocks**: Move large files to `/scratch` or `/project`

**Quick fixes:**
```bash
# Find large files
du -h --max-depth=1 ~/ | sort -h

# Find many files in directory
find ~/dir -type f | wc -l

# Compress directory
tar -czvf archive.tar.gz ./directory/
rm -rf ./directory/

# Move to scratch
mv large_data /scratch/midway3/$USER/
```

### Best Practices

1. **Store code** in `/home` (small, backed up)
2. **Store data** in `/project/jevans/` (large, shared, backed up)
3. **Store temporary files** in `/scratch` (large, temporary, NOT backed up)
4. **Use job-local scratch** (`$TMPDIR`) for I/O intensive operations
5. **Clean up scratch** regularly - data can be purged without warning
6. **Compress data files** to save space and file count
7. **Delete old job outputs** you no longer need

### Requesting More Storage

- Through Cluster Partnership Program
- As part of Research II Allocation
- Contact help@rcc.uchicago.edu for storage purchase options

> **Source:** RCC User Guide - Storage (https://docs.rcc.uchicago.edu/storage/main/), accessed January 2026

---

## Appendix E: Software Modules Reference

### Module System Overview

RCC uses Environment Modules to manage software. Users must explicitly load modules to access installed packages.

### Essential Module Commands

| Command | Description |
|---------|-------------|
| `module avail` | List all available modules |
| `module avail python` | Search for modules matching "python" |
| `module load <name>` | Load a module |
| `module load <name>/<version>` | Load specific version |
| `module unload <name>` | Unload a module |
| `module list` | Show currently loaded modules |
| `module purge` | Unload all modules |
| `module show <name>` | Show module details |

### AMD Partition Modules

For AMD nodes on Midway3, load AMD-specific modules:
```bash
module use /software/modulefiles-amd
module avail  # Now shows AMD-optimized modules
```

### Commonly Used Modules

| Category | Module Examples |
|----------|----------------|
| **Python** | `python/anaconda-2022.05`, `python/3.11.9`, `python/miniforge-25.3.0` |
| **R** | `R/4.2.0` |
| **MATLAB** | `matlab/2023a` |
| **Deep Learning** | `tensorflow`, `pytorch`, `cuda` |
| **Scientific** | `gromacs`, `lammps`, `gaussian` |
| **Data Processing** | `spark`, `hadoop` |
| **Compilers** | `gcc`, `intel`, `cmake` |

### Handling Module Conflicts

If modules conflict:
```bash
module unload conflicting-module
module load desired-module
```

Force load (use cautiously):
```bash
module load -f <module>
```

### Installing Custom Software

1. Load base modules first
2. Install additional packages via pip/conda
3. Store in `/project/jevans/` not `/home` to avoid quota issues

> **Source:** RCC User Guide - Software (https://docs.rcc.uchicago.edu/software/), accessed January 2026

---

## Appendix F: Python and Jupyter Configuration

### Available Python Distributions

| Module | Description | Recommendation |
|--------|-------------|----------------|
| `python/3.11.9` | Standard Python | Most research |
| `python/miniforge-25.3.0` | Conda-forge based | Scientific computing (recommended) |
| `python/anaconda-2022.05` | Anaconda distribution | Legacy workflows only |

**Note:** Anaconda has commercial licensing restrictions for large organizations. Miniforge is recommended.

### Creating Virtual Environments

**Store environments in project space** (not home) to avoid quota issues:

```bash
# Load Python
module load python/miniforge-25.3.0

# Create environment in project space
conda create --prefix=/project/jevans/$USER/envs/myenv python=3.11

# Activate using source (NOT conda activate)
source activate /project/jevans/$USER/envs/myenv
```

**Create convenient symlink:**
```bash
mkdir -p ~/.conda/envs
ln -s /project/jevans/$USER/envs/myenv ~/.conda/envs/myenv
source activate myenv  # Now works with short name
```

### Pre-configured Environments (Miniforge)

| Environment | Command | Purpose |
|-------------|---------|---------|
| sci | `source activate sci` | Scientific computing, data analysis |
| ml | `source activate ml` | Deep learning, ML research |
| bio | `source activate bio` | Bioinformatics, genomics |
| geo | `source activate geo` | GIS, earth science |
| hpc | `source activate hpc` | Parallel/distributed computing |

### Jupyter Lab Setup

**Step 1: Start interactive session**
```bash
sinteractive --account=pi-jevans --partition=caslake --time=04:00:00
```

**Step 2: Get node IP and start Jupyter**
```bash
module load python/anaconda-2022.05
HOST_IP=$(/sbin/ip route get 8.8.8.8 | awk '{print $7;exit}')
jupyter lab --no-browser --port=18765 --ip=$HOST_IP
```

**Step 3:** Click the URL in the terminal output to open in browser.

### SSH Tunnel for Remote Access (without VPN)

On your local machine:
```bash
ssh -N -f -L 18765:<HOST_IP>:18765 <CNetID>@midway3.rcc.uchicago.edu
```

Then open: `http://127.0.0.1:18765/?token=...`

### Best Practices

1. **Never run `conda init`** on Midway - it corrupts shell environments
2. **Always use `source activate`** instead of `conda activate`
3. **Set cache directories** to project space:
   ```bash
   export CONDA_PKGS_DIRS=/project/jevans/$USER/conda/pkgs
   export USE_CONDA_CACHE=0
   ```
4. **Clean caches regularly:**
   ```bash
   conda clean --all
   ```
5. **Document environments:**
   ```bash
   conda env export --from-history > environment.yml
   ```

> **Source:** RCC User Guide - Python (https://docs.rcc.uchicago.edu/software/apps-and-envs/python/), accessed January 2026

---

## Appendix G: Data Sources Documentation

### OpenAlex

**What it is:** A fully open catalog of the global research system with 240M+ scholarly works.

**Data entities:**
- Works (articles, books, datasets, theses)
- Authors (disambiguated researcher profiles)
- Sources (journals, repositories)
- Institutions (universities, organizations)
- Topics (subject classifications)
- Publishers and Funders

**Access methods:**

1. **Local snapshot:** `/project/jevans/tip/data/openalex/` (October 2023)
2. **API:** Free, 100,000 requests/day, no authentication
   ```
   Base URL: https://api.openalex.org
   Example: https://api.openalex.org/works?filter=authorships.author.id:A123456789
   ```
3. **Monthly snapshots:** Available for bulk download

**Advantages over alternatives:**
- ~2x coverage of Scopus/Web of Science
- Better coverage of non-English works and Global South research
- CC0 license (fully open)

**Official documentation:** https://docs.openalex.org

> **Source:** OpenAlex Documentation (https://docs.openalex.org), accessed January 2026

---

### Web of Science

**What it is:** Clarivate's curated database of high-impact research publications.

**Local data:** `/project/jevans/tip/data/wos_2023/` (XML and Parquet)

**API access:** Requires institutional license

| API | Description | Access |
|-----|-------------|--------|
| Starter API | Basic metadata, DOI lookup | Free tier: 50 requests/day |
| Expanded API | Full metadata, citations | Paid license |
| Researcher API | Author records | Paid license |
| Journals API | Journal Citation Reports | Paid license |

**Developer portal:** https://developer.clarivate.com/apis

> **Source:** Clarivate Developer Portal (https://developer.clarivate.com/apis/wos), accessed January 2026

---

### IRIS UMETRICS

**What it is:** Administrative data from research universities tracking research investments, personnel, and outputs.

**Data scope (2024 release):**
- 580,000+ funded awards ($192 billion)
- 1.2M+ vendor payments ($35 billion)
- ~985,000 employee wage records

**Data types:**
- University HR, sponsored projects, procurement data
- Linked to publications, patents, dissertations
- Integration with Census LEHD and Longitudinal Business Database

**Access requirements:**
1. Data Use Agreement (DUA) with University of Michigan
2. IRB determination letter from your institution
3. Completion of required training
4. Seat fee (for non-IRIS member institutions)

**Access method:** Virtual Data Enclave (VDE) with annual renewal

**Application process:**
1. Contact IRIS at University of Michigan
2. Submit research project description
3. Obtain IRB approval
4. Complete training
5. Pay seat fee if applicable

**Official website:** https://iris.isr.umich.edu/

> **Source:** IRIS - Institute for Research on Innovation & Science (https://iris.isr.umich.edu/), accessed January 2026

---

### PubMed Knowledge Graph (PKG)

**What it is:** Comprehensive knowledge graph connecting biomedical papers, patents, and clinical trials.

**PKG 2.0 scope:**
- 36M+ papers
- 1.3M patents
- 0.48M clinical trials
- 482M biomedical entity linkages
- 19M citation linkages
- 7M project linkages

**Local data:** `/project/jevans/PKG_v2_2023`

**Data structure:**
- A-prefix tables: Original PubMed data
- B-prefix tables: External source data
- C-prefix tables: Core PKG integrated data

**Use cases:**
- Citation analysis
- Knowledge reasoning
- Biomedical NLP
- Drug discovery research

**Latest version:** PKG24S4 (July 2025), based on PubMed 2025 Baseline

**Official website:** https://pubmedkg.github.io/

**License:** MIT

> **Source:** PubMed Knowledge Graph (https://pubmedkg.github.io/), Scientific Data publication, accessed January 2026

---

### Academic Torrents

**What it is:** Distributed repository for research datasets using BitTorrent infrastructure.

**Statistics:**
- 298+ TB of research data
- Used by MIT, Stanford, Berkeley, CMU, and others

**Categories:**
- Datasets
- Papers
- Courses
- Collections

**Use for Reddit data:**
Download Reddit dumps via aria2c on AWS EC2, then transfer to Midway via scp.

**Website:** https://academictorrents.com/

> **Source:** Academic Torrents (https://academictorrents.com/), accessed January 2026

---

## Appendix H: SSCS Acropolis Reference

### Overview

SSCS Acropolis is a dedicated cluster for social science research, separate from Midway.

**Key differences from Midway:**
- Uses PBS/Torque job scheduler (not SLURM)
- Files NOT shared with Midway
- Different resource limits

### Connection

```bash
ssh <CNetID>@acropolis.uchicago.edu
```

For graphical applications, use X11 forwarding (not VNC).

### System Specifications

| Component | Specification |
|-----------|---------------|
| Head Node | Dell R920, Intel Xeon E7-8891 v2 @ 3.20GHz |
| Cores | 40 cores (80 threads) |
| Max Resources | 350 cores, 1 TB memory per user |

### Storage

| Location | Description |
|----------|-------------|
| `/home` | Home directories (10GbE network) |
| `/share` | Shared storage (10GbE network) |
| `/scratch` | Local SSD on each compute node |

### PBS/Torque Commands

| PBS Command | SLURM Equivalent | Description |
|-------------|-----------------|-------------|
| `qsub script.pbs` | `sbatch script.sbatch` | Submit batch job |
| `showq` | `squeue` | View job queue |
| `checkjob <jobid>` | `scontrol show job` | Job details |
| `qdel <jobid>` | `scancel <jobid>` | Cancel job |

### Example PBS Script

```bash
#!/bin/bash
#PBS -N myjob
#PBS -l nodes=1:ppn=8
#PBS -l walltime=04:00:00
#PBS -l mem=32gb

cd $PBS_O_WORKDIR
module load matlab
matlab -nodisplay < script.m
```

**Official documentation:** https://sscs.uchicago.edu/category/faq/cluster

> **Source:** SSCS Acropolis FAQ (https://sscs.uchicago.edu/category/faq/cluster), accessed January 2026

---

## Appendix I: RCC Allocations Guide

### Allocation Types

| Type | Availability | Duration | SU Limit | Storage |
|------|-------------|----------|----------|---------|
| **Startup** | Anytime (once per PI) | 1 year | 10,000 SUs | 500 GB |
| **Education** | Anytime (multiple) | Class duration | Flexible | - |
| **Special** | Anytime | Year-cycle (Oct 1 - Sep 30) | Flexible | - |
| **Supplemental** | Twice annually | Year-cycle | 300K-500K SUs | - |
| **Research I** | Anytime | Year-cycle | 50K-100K SUs | 500 GB |
| **Research II** | September & March only | Year-cycle | 1M-2M SUs (Sep), 500K-1M (Mar) | 1024 GB |

**Note:** Limits vary based on Cluster Partnership Program membership.

### Service Unit (SU) Definition

**1 SU = 1 core-hour** on the Midway Cluster

### Checking Allocation Status

```bash
accounts allocations  # View current allocation
accounts balance      # View remaining balance
rcchelp usage -accounts pi-jevans -byjob | grep <cnetid>  # Detailed usage
```

### KLab Annual Renewal

- **When:** September each year
- **What:** Apply to renew allocation for pi-jevans account
- **Who:** PI or designated lab member

### Not Charged Against Allocation

- Usage of `jevans` and `jevans-gpu` partitions (lab-owned)
- Debug QoS sessions (15 minutes, 4 cores)

### Requesting More Resources

- **Additional storage:** Contact help@rcc.uchicago.edu
- **More compute:** Apply for Research II or Special Allocation
- **General support:** (773) 795-2667

**Official information:** https://rcc.uchicago.edu/accounts-allocations/request-allocation

> **Source:** RCC Accounts & Allocations (https://rcc.uchicago.edu/accounts-allocations/request-allocation), accessed January 2026

---

## Citation Index

All sources referenced in this document, organized alphabetically:

| Source | URL | Section(s) |
|--------|-----|------------|
| Academic Torrents | https://academictorrents.com/ | Data Sets, Appendix G |
| Clarivate Developer Portal | https://developer.clarivate.com/apis/wos | Appendix G |
| IRIS UMETRICS | https://iris.isr.umich.edu/ | Data Sets, Appendix G |
| NVIDIA H100 Datasheet | https://www.nvidia.com/en-us/data-center/h100/ | Appendix C |
| OpenAlex Documentation | https://docs.openalex.org | Data Sets, Appendix G |
| PubMed Knowledge Graph | https://pubmedkg.github.io/ | Data Sets, Appendix G |
| RCC Accounts & Allocations | https://rcc.uchicago.edu/accounts-allocations/request-allocation | Midway3, Appendix I |
| RCC User Guide - Partitions | https://docs.rcc.uchicago.edu/partitions/ | Midway3, Appendix A |
| RCC User Guide - Python | https://docs.rcc.uchicago.edu/software/apps-and-envs/python/ | Appendix F |
| RCC User Guide - sbatch | https://docs.rcc.uchicago.edu/slurm/sbatch/ | Appendix B |
| RCC User Guide - sinteractive | https://docs.rcc.uchicago.edu/slurm/sinteractive/ | Appendix B |
| RCC User Guide - Software | https://docs.rcc.uchicago.edu/software/ | Software Tips, Appendix E |
| RCC User Guide - SSH | https://docs.rcc.uchicago.edu/ssh/main | Midway3, Appendix A |
| RCC User Guide - Storage | https://docs.rcc.uchicago.edu/storage/main/ | Storage, Appendix D |
| SSCS Acropolis FAQ | https://sscs.uchicago.edu/category/faq/cluster | Acropolis, Appendix H |

---

**Document Information:**
- Original: KLab Computing Manual (Living Document)
- Enriched Edition: January 2026
- Purpose: Comprehensive reference for Chorus knowledge base
- Maintenance: Update appendices as official documentation changes
