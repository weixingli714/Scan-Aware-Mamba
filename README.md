## For the paper **"Scan-Aware Mamba for Fine-grained Type-B Aortic Dissection Segmentation with Quantitative Anatomical Assessment"**

### FineTBAD Dataset

To support fine-grained TBAD segmentation, we construct **FineTBAD**, a multi-source CTA dataset comprising **125 cases** with unified annotations of up to **22 foreground classes**. The annotation protocol covers anatomically subdivided aortic regions, segment-specific false lumina, iliac and femoral arteries, and major supra-aortic and visceral branch arteries. FineTBAD is designed to support both detailed vascular segmentation and subsequent quantitative anatomical assessment.

<p align="center">
  <img src="assets/2.png" width="95%">
</p>


### Scan-Aware Mamba (SAMamba)

The proposed **Scan-Aware Mamba** follows a U-shaped 3D segmentation architecture for fine-grained TBAD analysis. The encoder combines convolutional feature extraction with **Scan-Aware Mamba (SAMamba) blocks**, where **Multi-view Scanning (MvS)** captures complementary anatomical context from different orthogonal views and the **Scan-Aware Gate (SAG)** adaptively integrates the resulting representations. In the decoder, the proposed **Vascular Cross-scale Attention Fusion (VCAF)** module aligns and integrates multi-scale encoder features to improve the reconstruction of vascular structures with substantial scale variation.

<p align="center">
  <img src="assets/1.png" width="95%">
</p>


### Implementation Framework

All experiments are implemented within the **nnUNetv2** framework, which provides a unified pipeline for preprocessing, data loading, training, inference, and postprocessing. To ensure consistent experimental settings, the proposed method, comparison methods, and ablation variants are organized as **nnUNetv2 trainer variants** whenever applicable. For architectures available in **MONAI**, the official network implementations are used as the basis for re-implementation and evaluation.

This design keeps preprocessing, optimization, inference, and evaluation procedures consistent across different methods, while allowing the architectural components of **SAMamba**, **SAG**, and **VCAF** to be evaluated within the same training framework.


## Main Developers

- **Weixing Li**<sup>1,2</sup>
- **Yu Sun**<sup>1,3,4</sup>
- **Wei Qian**<sup>1,2</sup>
- **Libo Zhang**<sup>3,4</sup>
- **Shouliang Qi**<sup>1,2</sup>

<sup>1</sup> College of Medicine and Biological Information Engineering, Northeastern University, Shenyang, China <br/>
<sup>2</sup> Key Laboratory of Intelligent Computing in Medical Image, Ministry of Education, Northeastern University, Shenyang, China <br/>
<sup>3</sup> Department of Radiology, General Hospital of Northern Theater Command, Shenyang, China <br/>
<sup>4</sup> Key Laboratory of Cardiovascular Imaging and Research of Liaoning Province, Shenyang, China <br/>


## License

This project is licensed under the [Apache License 2.0](LICENSE).
