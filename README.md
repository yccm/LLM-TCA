# LLM-Driven CATE for Text Confounding

PyTorch Implementation on Paper [NeurIPS 2025] [LLM-Driven Treatment Effect Estimation Under Inference Time Text Confounding](https://arxiv.org/abs/2507.02843)

## Introduction

Estimating treatment effects is crucial for personalized decision-making in medicine, but this task faces unique challenges in clinical practice. At training time, models for estimating treatment effects are typically trained on well-structured medical datasets that contain detailed patient information. However, at inference time, predictions are often made using textual descriptions (e.g., descriptions with self-reported symptoms), which are incomplete representations of the original patient information. In this work, we propose a novel framework for estimating treatment effects that explicitly accounts for inference time text confounding. Our framework leverages large language models (LLMs) together with a custom doubly robust learner to mitigate biases caused by the inference time text confounding. 



### Installation:
`python 3.10.16
pytorch 2.5.1`


### Getting started:

#### Prerequisites:

#### Training Example




## Bibtex
``` 
@inproceedings{ma2025llm,
  title={LLM-Driven Treatment Effect Estimation Under Inference Time Text Confounding},
  author={Ma, Yuchen and Frauen, Dennis and Schweisthal, Jonas and Feuerriegel, Stefan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```




