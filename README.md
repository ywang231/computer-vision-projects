# 🔧 Configuration Overview

This project utilizes **AWS Celebrity Face Rekognition (ACFR)** alongside **Stable Diffusion** for facial recognition and image generation tasks.

- Due to **incompatibility between ACFR and Google Colab**, ACFR-related code was executed in a **VS Code + Jupyter plugin environment**.
- **Google Colab** was used to leverage **high-performance GPUs** for Stable Diffusion and fine-tuning tasks.
- To use the **AWS Rekognition API**, please supply your own API key and secret:
  - 🔐 **Note**: The placeholders in the code are not valid.

---

# 📁 Project Structure

| Folder/File | Description |
|-------------|-------------|
| [`heads`](https://github.com/EricW1118/ComVisionProject/tree/main/heads) | Head photos from Kaggle dataset |
| [`headsets`](https://github.com/EricW1118/ComVisionProject/tree/main/headsets) | Two head images randomly selected per celebrity from the `heads` folder |
| [`dev.csv`](https://github.com/EricW1118/ComVisionProject/blob/main/dev.csv) | Original dataset with image URLs, names, and face regions |
| [`dev`](https://github.com/EricW1118/ComVisionProject/tree/main/dev) | Downloaded images from `dev.csv` using `download_preprocess.ipynb` |
| [`download_preprocess.ipynb`](https://github.com/EricW1118/ComVisionProject/blob/main/download_preprocess.ipynb) | Downloads and preprocesses images from `dev.csv` |
| [`cv_main.ipynb`](https://github.com/EricW1118/ComVisionProject/blob/main/cv_main.ipynb) | Applies basic obfuscation methods: Gaussian noise, eye patches, lines, and leopard spots |
| [`config.csv`](https://github.com/EricW1118/ComVisionProject/blob/main/config.csv) | Recognition rates for 300 celebrities after obfuscation (used for graphs and comparisons) |
| [`Encryption_Results.py`](https://github.com/EricW1118/ComVisionProject/blob/main/Encryption_Results.py) | Encryption method for testing ACFR recognition (run in AWS) |
| [`Encrypt_config.csv`](https://github.com/EricW1118/ComVisionProject/blob/main/Encrypt_config.csv) | Recognition results from `Encryption_Results.py` |
| [`StableDiff.ipynb`](https://github.com/EricW1118/ComVisionProject/blob/main/StableDiff.ipynb) | Generates images using Stable Diffusion’s Image2Image pipeline (no fine-tuning) |
| [`Fine_tuning.ipynb`](https://github.com/EricW1118/ComVisionProject/blob/main/Fine_tuning.ipynb) | Fine-tunes Stable Diffusion using the Text2Image pipeline |
| [`concepts_list.json`](https://github.com/EricW1118/ComVisionProject/blob/main/concepts_list.json) | Contains class/instance prompts used in fine-tuning |
| [`inpaint.ipynb`](https://github.com/EricW1118/ComVisionProject/blob/main/inpaint.ipynb) | Inpainting using Stable Diffusion (**incomplete**, future work) |
| [`commonfuns.py`](https://github.com/EricW1118/ComVisionProject/blob/main/commonfuns.py) | Frequently used utility functions across notebooks |
| [`requirements.text`](https://github.com/EricW1118/ComVisionProject/blob/main/requirements.text) | Environment dependencies for fine-tuning code |
| [`train_dreambooth.py`](https://github.com/EricW1118/ComVisionProject/blob/main/train_dreambooth.py) | Adapted script for training with **DreamBooth** (not used directly due to compilation issues) |
