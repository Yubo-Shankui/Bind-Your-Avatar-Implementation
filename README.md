
<h1 align='center'>Bind-Your-Avatar: Multi-Talking-Character Video Generation with Dynamic 3D-mask-based Embedding Router</h1>








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

## ⚙️ Environments

We recommend the requirements as follows. 

```bash
conda create -n bindyouravatar python=3.11.0
conda activate bindyouravatar
pip install -r requirements.txt
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
