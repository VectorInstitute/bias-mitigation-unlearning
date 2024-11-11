## Introduction
This repository adopts 2 machine unlearning based methods: **Negation via Task Vectors** and **PCGU**, for mitigating social biases in language models. Our paper [Can Machine Unlearning Reduce Social Bias in Language Models?](https://aclanthology.org/2024.emnlp-industry.71) (Dige et al., EMNLP 2024) has been published in the Industry Track of EMNLP 2024.

## Running scripts
1. Install required packages:
```bash
python -m pip install -r requirements.txt
```

2. Instructions for running scripts are available in the respective directories for each method.

## Acknowledgements
This work has resulted from a larger collaborative initiative involving the Vector Institute and its industry partners. The authors extend their appreciation to Tahniat Khan, the project manager, for her efforts in coordinating this project. We also express our thanks to Deval Pandya, Vice President of AI Engineering at the Vector Institute, for his valuable support.

The authors would like to acknowledge the leaders at Ernst & Young (EY) for their exceptional support and commitment to advancing artificial intelligence research. Special thanks to Mario Schlener, Managing Partner for Risk Consulting Canada, whose strategic vision exemplifies EY's dedication to fostering innovation and thought leadership in the industry. We also recognize the expert oversight of Yara Elias, Kiranjot Dhillon, and Rasoul Shahsavarifar from AI Risk Canada, whose contributions were integral to the project's success. This partnership not only reflects EY's investment in AI but also sets a foundation for continued research collaboration and driving progress in the field.

## Citation
To cite our work:

```
@inproceedings{dige-etal-2024-machine,
    title = "Can Machine Unlearning Reduce Social Bias in Language Models?",
    author = "Dige, Omkar  and
      Arneja, Diljot  and
      Yau, Tsz Fung  and
      Zhang, Qixuan  and
      Bolandraftar, Mohammad  and
      Zhu, Xiaodan  and
      Khattak, Faiza Khan",
    editor = "Dernoncourt, Franck  and
      Preo{\c{t}}iuc-Pietro, Daniel  and
      Shimorina, Anastasia",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track",
    month = nov,
    year = "2024",
    address = "Miami, Florida, US",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-industry.71",
    pages = "954--969",
    abstract = "Mitigating bias in language models (LMs) has become a critical problem due to the widespread deployment of LMs in the industry and customer-facing applications. Numerous approaches revolve around data pre-processing and subsequent fine-tuning of language models, tasks that can be both time-consuming and computationally demanding. As alternatives, machine unlearning techniques are being explored, yet there is a notable lack of comparative studies evaluating the effectiveness of these methods. In this work, we explore the effectiveness of two machine unlearning methods: Partitioned Contrastive Gradient Unlearning (PCGU) applied on decoder models, and Negation via Task Vector, and compare them with Direct Preference Optimization (DPO) to reduce social biases in open-source LMs such as LLaMA-2 and OPT. We also implement distributed PCGU for large models. It is empirically shown, through quantitative and qualitative analyses, that negation via Task Vector method outperforms PCGU and is comparable to DPO in debiasing models with minimum deterioration in model performance and perplexity. Negation via Task Vector reduces the bias score by 25.5{\%} for LLaMA-2 and achieves bias reduction of up to 40{\%} for OPT models. Moreover, it can be easily tuned to balance the trade-off between bias reduction and generation quality, unlike DPO.",
}
```
