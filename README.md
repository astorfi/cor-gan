# COR-GAN: Correlation-Capturing Convolutional Neural Networks for Generating Synthetic Healthcare Records

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/astorfi/cor-gan.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/astorfi/cor-gan/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/astorfi/cor-gan.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/astorfi/cor-gan/alerts/)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cor-gan-correlation-capturing-convolutional/synthetic-data-generation-on-uci-epileptic)](https://paperswithcode.com/sota/synthetic-data-generation-on-uci-epileptic?p=cor-gan-correlation-capturing-convolutional)


This repository contains an implementation of "COR-GAN:
Correlation-Capturing Convolutional Neural Networks for Generating
Synthetic Healthcare Records". This is not an officially supported
Google product.


For the following reason, this implementation *does not contain* the ``full implementation`` of the code:
* The privacy restrictions regarding this work.

For a detailed description of the architecture please read [our paper](https://arxiv.org/abs/2001.09346). Using the code of this repository is allowed with **proper attribution**: Please cite the paper if you use the code from this repository in your work.

## Bibtex

    @article{torfi2020cor,
    title={COR-GAN: Correlation-Capturing Convolutional Neural Networks for Generating Synthetic Healthcare Records},
    author={Torfi, Amirsina and Fox, Edward A},
    journal={arXiv preprint arXiv:2001.09346},
    year={2020}
    }


Table of contents
=================

<!--ts-->
   * [Paper Summary](#paper-summary)
   * [Aspects of The Work](#aspects-of-the-work)
      * [Synthetic Data Generation](#Synthetic-Data-Generation)
      * [Privacy](#Privacy)
      * [Data Fidelity](#data-fidelity)
   * [Running the Code](#Running-the-Code)
      * [Prerequisites](#Prerequisites)
      * [Datasets](#Datasets)
      * [Training](#Training)
   * [Collaborators](#Collaborators)
<!--te-->


## Paper Summary

<details>
<summary>Abstract</summary>

*Deep learning models have demonstrated high-quality performance in areas such as image classification and speech processing.
However, creating a deep learning model using electronic health record (EHR) data, requires addressing particular privacy challenges that are unique to researchers in this domain. This matter focuses attention on generating realistic synthetic data while ensuring privacy.
In this paper, we propose a novel framework called correlation-capturing Generative Adversarial Network (corGAN), to generate synthetic healthcare records. In corGAN we utilize Convolutional Neural Networks to capture the correlations between adjacent medical features in the data representation space by combining Convolutional Generative Adversarial Networks and Convolutional Autoencoders.
To demonstrate the model fidelity, we show that corGAN generates synthetic data with performance similar to that of real data in various Machine Learning settings such as classification and prediction. We also give a privacy assessment and report on statistical analysis regarding realistic characteristics of the synthetic data.*

</details>

<!-- <br>[⬆ Back to top](#contents) -->

<!-- ### Abstract

*Deep learning models have demonstrated high-quality performance in areas such as image classification and speech processing.
However, creating a deep learning model using electronic health record (EHR) data, requires addressing particular privacy challenges that are unique to researchers in this domain. This matter focuses attention on generating realistic synthetic data while ensuring privacy.
In this paper, we propose a novel framework called correlation-capturing Generative Adversarial Network (corGAN), to generate synthetic healthcare records. In corGAN we utilize Convolutional Neural Networks to capture the correlations between adjacent medical features in the data representation space by combining Convolutional Generative Adversarial Networks and Convolutional Autoencoders.
To demonstrate the model fidelity, we show that corGAN generates synthetic data with performance similar to that of real data in various Machine Learning settings such as classification and prediction. We also give a privacy assessment and report on statistical analysis regarding realistic characteristics of the synthetic data.* -->

<details>
<summary>Motivation</summary>

* Synthetic records helping stakeholders to *share and work* on data without privacy hurdles
* Despite advances in *Synthetic Data Generation~(SDG)*. Research efforts mostly restricted to *limited use cases*
* *Lack of clarity* regarding the synthetic data being realistic; and factors contributing to realism
* Majority of methods for *supervised settings*
* Lack of clarity in *measuring privacy*, and privacy guarantees
* EHRs are *discrete* in nature. But most research is on *continuous* data

</details>

<details>
<summary>Contribution</summary>

* We propose an efficient architecture to generate synthetic healthcare records using **Convolutional GANs** and **Convolutional Autoencoders}~(CAs)** which we call ``corGAN``. We demonstrate that corGAN can effectively generate both *discrete* and *continuous* synthetic records.
* We demonstrate the effectiveness of utilizing Convolutional Neural Networks~(CNNs) as opposed to Multilayer Perceptrons to capture inter-correlation between features.
* We show that corGAN can generate realistic synthetic data that performs similarly to real data on classification tasks, according to  our analysis and assessments.
* We report on a **privacy assessment** of the model and demonstrate that corGAN provides an acceptable level of privacy, by varying the amount of synthetically generated data and amount of data known to an adversary.

</details>


## Aspects of The Work

### Synthetic Data Generation

<details>
<summary>Details</summary>

The discrete input **X** represents the source EHR data; **z** is the random distribution for the generator **G**; **G** is the employed neural network architecture; **Dec(G(z))}** refers to the decoding function which is used to transform the generator **G** continuous output to their equivalent discrete values. The discriminator **D** attempts to distinguish real input **X** from the discrete synthetic output **Dec(G(z))}**. For the generator and the discriminator, a 1-Dimensional Convolutional GAN architecture is utilized.

</details>

<img src="https://github.com/astorfi/cor-gan/blob/master/imgs/proposedarch.png" width="50%" height="50%" />


### Privacy - Membership Inference Attack

<details>
<summary>Details</summary>

We utilize the **Membership Inference (MI)** attack as an approach to measure the privacy. *Membership Inference~(MI)* refers to determining whether a given record generated by a known machine learning model was used as part of the training data.
The membership inference problem is basically the well-known problem of *presence disclosure* of an individual.
If the adversary has complete access to the records of a particular patient and can recognize their employment in the model training, that is an indication of information leakage, as it can jeopardize the whole dataset privacy or at least the particular patient's private information.
Here, we will assume the adversary *has the synthetically generated data as well as a portion of the compromised real data*.
</details>

<img src="https://github.com/astorfi/cor-gan/blob/master/imgs/membershipattack.png" width="50%" height="50%" />

### Data Fidelity

<details>
<summary>Details</summary>

**Binary Classification:** We use this metric for our experiments  with  continuous  data. To  empirically  verify  the quality  of  the  synthetic  data,  we  consider  two  different settings. ``(A)`` Train and test the predictive models on the real data. ``(B)``train the predictive model on synthetic data and test it on the real data. **If the model evaluated in setting B, represents competitive results with the same model performed in setting (A)**, we can conclude the synthetic data has good predictive modeling similar to the real data.

</details>

<img src="https://github.com/astorfi/cor-gan/blob/master/imgs/datafidelity.png" width="50%" height="50%" />


## Running the Code

### Prerequisites

* Pytorch ``1.4``
* CUDA [strongly recommended]

**NOTE:** PyTorch does a pretty good job in installing required packages but you should have installed CUDA according to PyTorch requirements.
Please refer to [this link](https://pytorch.org/) for further information.

### Datasets

You need to download and process the following datasets as due to privacy restrictions we cannot provide the data here.

* MIMIC-III dataset: https://mimic.physionet.org/ [implementation with this dataset is included]
* UCI Epileptic Seizure Recognition dataset: https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition [implementation with this dataset is NOT included]

One good source code for processing MIMIC-III can be found [here](https://github.com/mp2893/medgan).


### Training

To check the implementation refer to the folder ``Generative`` and you will see the following implementations:

* corGAN: The implementation of the paper with concolutional autoencoders and regular multi-layer perceptions for discriminator and generator.
* medGAN: The implementation of the [medGAN paper](https://arxiv.org/abs/1703.06490) as one of the baselines. This is a reimplemetation of the medGAN in ``PyTorch`` as we could not fully reproduce their results with [their code](https://github.com/mp2893/medgan). Furthermore, **PyTorch is more flexible compare to TensorFlow for research!**
* VAE: One of the the baselines.

## Collaborators

| [<img src="https://github.com/astorfi.png" width="100px;"/>](https://github.com/astorfi)<br/> [<sub>Amirsina Torfi</sub>](https://github.com/astorfi) | [<img src="https://github.com/mohibeyki.png" width="100px;"/>](https://github.com/mohibeyki)<br/> [<sub>Mohammadreza Beyki</sub>](https://github.com/mohibeyki) |
| --- | --- |

<!-- ## Credit

This research conducted at [Virginia Tech](https://vt.edu/) under the supervision of [Dr. Edward A. Fox](http://fox.cs.vt.edu/foxinfo.html). -->
