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
