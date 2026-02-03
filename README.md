# 📚 Machine Learning Portfolio

Welcome to my **Machine Learning Portfolio**! This repository showcases a diverse range of projects where I've applied machine learning techniques to tackle various challenges. Let's dive into what you'll find here.

## Table of Contents

- [Overview](#overview)
- [License](#license)

## Overview

🌟 This portfolio highlights my journey in the field of machine learning, featuring projects that demonstrate a range of algorithms, data processing techniques, and model evaluation strategies. Each project is an opportunity to see theory applied practically.

## Installation

🛠️ To run any project from this portfolio, ensure Python is installed along with necessary libraries. Use a virtual environment for managing dependencies smoothly.

```bash
pip install -r requirements.txt
```

## License

📄 This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Thank you for exploring my **Machine Learning Portfolio**! I hope it provides insights into how I approach machine learning challenges. If you have questions or feedback, feel free to reach out at <dillondalton0952362@gmail.com>.

Happy exploring! 🚀

```mermaid
flowchart TD
  P[Planner\n- short plan\n- propose multiple retrieval queries] --> L[Librarian\n- run each query on FAISS\n- consolidate best chunks]

  L -. retrieval .-> IDX[(FAISS index)]
  IDX -. chunks .-> L

  L --> EP[[Evidence package]]
  EP --> W[Writer\n- draft using only evidence package\n- add citations: [row=...]\n- note limitations if evidence is weak]

  W --> DRAFT[[Draft]]
  DRAFT --> R[Reviewer\n- check draft vs evidence\n- flag unsupported claims\n- flag missing citations/trade-offs\n- output required edits]

  R --> EDITS[[Required edits]]
  EDITS --> F[Finalizer\n- apply edits\n- preserve citations\n- grounded final response]

  F --> OUT[Final response]
```
