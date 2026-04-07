
<h1 align='center'>Bind-Your-Avatar: Multi-Talking-Character Video Generation with Dynamic 3D-mask-based Embedding Router</h1>

<div align='center'>
  <a href='https://github.com/Yubo-Shankui' target='_blank'>Yubo Huang</a><sup>1</sup>&emsp;
  <a href='https://github.com/WJohnnyW' target='_blank'>Weiqiang Wang</a><sup>2</sup>&emsp;
  <a href='https://scholar.google.com/citations?user=bEZurKQAAAAJ&hl=en' target='_blank'>Sirui Zhao</a><sup>1✉️</sup>&emsp;
  <a href='http://staff.ustc.edu.cn/~tongxu/index_zh.html' target='_blank'>Tong Xu</a><sup>1</sup>&emsp;
  <a href='http://home.ustc.edu.cn/~ll0825/' target='_blank'>Lin Liu</a><sup>1✉️</sup>&emsp;
  <a href='http://staff.ustc.edu.cn/~cheneh/' target='_blank'>Enhong Chen</a><sup>1✉️</sup>
</div>

<div align='center'>
  <sup>1</sup>University of Science and Technology of China &emsp; <sup>2</sup>Monash University  
</div>

<div align='center'>
  <sup>✉️</sup> Corresponding Author
</div>

<br>
<div align='center'>
    <a href='https://yubo-shankui.github.io/bind-your-avatar/'><img src='https://img.shields.io/badge/Project-HomePage-Green'></a>
    <a href='https://arxiv.org/pdf/2506.19833'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

<br>
</div> 


## 📸 Showcase

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/f978ed51-429e-40ef-a231-c64cccc75ae6" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/0d171ce3-5ad4-4826-b59d-c7c6f76fb954" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/0eb1ad16-6732-44e8-9ccf-df8f80db4362" width="100%" controls autoplay loop></video>
     </td>
  </tr>
        <td>
          <video src="https://github.com/user-attachments/assets/5012462e-30e3-4fff-8a2b-a43af0aa6ef4" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/3badcf4b-3867-4792-ab58-2081ae56901b" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/cfcdf8f3-fa08-444a-b3c4-50f91f036a4e" width="100%" controls autoplay loop></video>
     </td>
  </tr>
  <tr>
</table>






Visit our [project page](https://yubo-shankui.github.io/bind-your-avatar/) to view more cases.

## 📰 News
- `[2025.7.31]`  🔥 We released the inference code and part of the training pipeline for Bind-Your-Avatar — more to come soon!
- `[2025.6.25]` 🔥 We release the arXiv paper for Bind-Your-Avatar, and you can click [here](https://arxiv.org/abs/2506.19833) to see more details.  
- `[2025.6.25]` 🔥 **All code, datasets & benchmark** are coming soon!  

## ⚙️ Environments

We recommend the requirements as follows. 

```bash
conda create -n bindyouravatar python=3.11.0
conda activate bindyouravatar
pip install -r requirements.txt
```

### Download BindYourAvatar

The weights are available at [🤗HuggingFace](https://huggingface.co/hyb1124/Bindyouravatar), you can download it with the following commands.

```bash
# if you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type model \
hyb1124/Bindyouravatar \
--local-dir pretrained
```

## 🗝️ Inference 
We provide the inference scripts ```batch_infer.sh``` for simple testing. Run the command as examples: 

```bash
bash batch_infer.sh
```

## ⏰ Training

We have released the Stage 3 training script, while Stage 1 & 2 are coming soon — including multi-stage training pipelines and multi-ID video datasets.

You can reproduce our Stage 3 experiments by simply running:

```bash
# For stage 3
bash train.sh
```

## 📝 Citation

If you find our work useful for your research, please consider citing the paper:

```
@misc{huang2025bindyouravatarmultitalkingcharactervideogeneration,
      title={Bind-Your-Avatar: Multi-Talking-Character Video Generation with Dynamic 3D-mask-based Embedding Router}, 
      author={Yubo Huang and Weiqiang Wang and Sirui Zhao and Tong Xu and Lin Liu and Enhong Chen},
      year={2025},
      eprint={2506.19833},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.19833}, 
}
```




