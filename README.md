# ComVisionProject

Dev.csv: The original dataset containing image URLs and head rectangle areas.
config.csv: The processed data with recognition rates after image analysis.
Image_download_preprocessing.ipynb: This Jupyter Notebook includes code for downloading images from the provided URLs, removing duplicate images, and cropping the head areas.
Image_experiments.ipynb: This Jupyter Notebook contains code for conducting image experiments, such as adding Gaussian noise, blocking eyes, adding leopard spots, and other modifications.
StableDiffusion.ipynb: This Jupyter Notebook focuses on using the Stable Diffusion model to generate synthetic images.

To utilize the AWS Celebrity Rekognition API, you will need to provide your own API key and API secret in the code. The current placeholders in the code are not valid. Please ensure that you have obtained the necessary API credentials from AWS.

Please note that AWS APIs are not compatible with Google Colab. To run this portion of the code, it is recommended to use Visual Studio Code with the Jupyter plugin. This setup will allow you to execute the code and interact with the AWS Celebrity Rekognition API.

Additionally, for the Stable Diffusion model, it is strongly recommended to use Google Colab's GPU to run it. Without GPU acceleration, the process may be significantly time-consuming.
