# Multi-Agent Reasoning for Cardiovascular Imaging Phenotype Analysis
**Official implementation of the MICCAI 2025 paper**  
>  Zhang W.\*, Qiao M.\*, Zang C., Niederer S., Matthews P., Bai W., Kainz B.  
> \* Equal contribution


## Overview

**MESHAgents** is a multi-agent reasoning framework designed for phenome-wide association studies (PheWAS) in cardiovascular imaging. It orchestrates a team of domain-specialized language model agents to automatically discover phenotype–factor associations through collaborative analysis and consensus building.

<p align="center">
  <img src="assets/meshagents_overview.png" alt="MESHAgents Framework" width="700"/>
</p>


## Features

- 🤝 Multi-agent consensus mechanism for robust phenotype discovery  
- 🧠 Agent memory and tool modules for statistical reasoning  
- 💬 Sequential discussion protocol inspired by clinical MDT panels  
- 🩺 Validated on UK Biobank (N=38,000+) for 9 disease phenotypes


## Installation

```bash
git clone https://github.com/your-username/MESHAgents.git
cd MESHAgents
conda env create -f environment.yml
conda activate meshagents
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{qiao2025meshagents,
  title     = {Multi-Agent Reasoning for Cardiovascular Imaging Phenotype Analysis},
  author    = {Zhang, Weitong and Qiao, Mengyun and Zang, Chengqi and Niederer, Steven and Matthews, Paul and Bai, Wenjia and Kainz, Bernhard},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year      = {2025}
}
```

## Acknowledgements
This work is supported by: EPSRC DeepGeM Grant (EP/W01842X/1), UKRI CDT in AI for Healthcare (EP/S023283/1), British Heart Foundation (RG/20/4/34803, NH/F/23/70013), National Institutes of Health (R01-HL152256), European Research Council (PREDICT-HF 864055, MIA-NORMAL 101083647, Deep4MI 884622), Edmond J. Safra Foundation, NIHR Senior Investigator Award, UK Dementia Research Institute

HPC resources were provided by the **Erlangen National High Performance Computing Center (NHR@FAU)** of the **Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).  

We also thank the UK Biobank for providing imaging data.
