
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
          <video src="https://github.com/user-attachments/assets/f26b1240-b22f-4f6c-9a24-64bf08848f15" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/9faaef7d-3cbf-4d8c-93ec-aada418ccfd3" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/10ec4363-1ea0-42f5-af79-d242be277289" width="100%" controls autoplay loop></video>
     </td>
  </tr>
        <td>
          <video src="https://github.com/user-attachments/assets/f0ba3e32-0881-4be6-b02f-0b7dffc0e39f" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/327cdf21-1042-4923-aa8b-730e8bd1dea4" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/4e0706f8-3634-47c4-816e-5eceb5c58280" width="100%" controls autoplay loop></video>
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

The weights of model will soon be available at 🤗HuggingFace.

## 🗝️ 3. Inference 
We provide the inference scripts ```batch_infer.sh``` for simple testing. Run the command as examples: 

```bash
bash batch_infer.sh
```

## ⏰ 4. Training

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




