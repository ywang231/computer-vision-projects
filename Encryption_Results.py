#!/usr/bin/env python
# coding: utf-8

# In[108]:


get_ipython().system('pip3 install opencv-python')
get_ipython().system('pip3 install boto3')


# In[109]:


# data processing
import pandas as pd
# AWS
import boto3
# OpenCV
import cv2
# file path
import os
# plot img
from matplotlib import pyplot as plt
# Numpy
import numpy as np
# seaborn
import seaborn as sns


# In[110]:


# Show progress bar
from tqdm import tqdm
# Apply progress bar to the DataFrame
tqdm.pandas()


# In[111]:


destination_path =  'Samples//'
# Create a dataframe to store the image info
config = pd.DataFrame(columns=['name', 'path','match_conf_original','match_conf_encryption'])
# # Get all the folders under heads folder
sub_folder = os.listdir(destination_path)
# Get all the images under each folder
for f in sub_folder:
    if f.endswith('.jpg') or f.endswith('.png'):
        path = os.path.join(f)
        names = f.split("_")[0]
        config.loc[len(config)] = {'name': names, 'path': path}
# show the first 5 rows of the dataframe
config.head


# In[112]:


from botocore.exceptions import ClientError

# Initialize the Amazon Rekognition client
AWS_ACCESS_KEY_ID ='AKIAZBOVGFWW3HWVAE6N'
AWS_SECRET_ACCESS_KEY ='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'
AWS_REGION = 'us-east-2'
rekognition_client = boto3.client('rekognition',
                                  region_name=AWS_REGION,
                                  aws_access_key_id=AWS_ACCESS_KEY_ID,
                                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

def amazon_celebrity_rekognition(image_path, rekognition_client = rekognition_client):
    session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
    )
    rekognition_client = session.client('rekognition')
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()

    # Call the Rekognition API to recognize celebrities
    response = rekognition_client.recognize_celebrities(Image={'Bytes': image_bytes})

    # Check if any celebrities were recognized
    if 'CelebrityFaces' in response and len(response['CelebrityFaces']) > 0:
    # Extract the name of the first recognized celebrity
        celebrity_name = response['CelebrityFaces'][0]['Name']
        Match_conf = response['CelebrityFaces'][0]['MatchConfidence']
        print(f"The recognized celebrity is: {celebrity_name}")
        config.loc[i, 'match_conf_original'] = Match_conf
        
    else:
        print("No celebrities were recognized in the image.")
        config.loc[i, 'match_conf_original'] = 0
        
def amazon_celebrity_rekognition_encryption(image_path, rekognition_client = rekognition_client):
    session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
    )
    rekognition_client = session.client('rekognition')
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()

    # Call the Rekognition API to recognize celebrities
    response = rekognition_client.recognize_celebrities(Image={'Bytes': image_bytes})

    # Check if any celebrities were recognized
    if 'CelebrityFaces' in response and len(response['CelebrityFaces']) > 0:
    # Extract the name of the first recognized celebrity
        celebrity_name = response['CelebrityFaces'][0]['Name']
        Match_conf = response['CelebrityFaces'][0]['MatchConfidence']
        print(f"The recognized celebrity is: {celebrity_name}")
        config.loc[i, 'match_conf_encryption'] = Match_conf
        
    else:
        print("No celebrities were recognized in the image.")
        config.loc[i, 'match_conf_encryption'] = 0


# In[113]:


row_count = config.shape[0]
for i in range(0, row_count):
    image_path = config.loc[i, 'path']
    image='Samples//'+ image_path
    celebrities = amazon_celebrity_rekognition(image)


# In[114]:


config.head


# In[116]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
def password_reader(password):
    password_values = []
    for alphabet in password:
        password_values.append(ord(alphabet))
    return password_values


def encryptor(image, password):
    # Get the shape of the image
    image = Image.open(image)
    image_array = np.asarray(image)
    encrypted_image = np.zeros_like(image_array,dtype=np.uint8)
    password_values = password_reader(password)


    encryptor_matrix = np.zeros_like(image_array, dtype=np.uint8)

    width = encryptor_matrix.shape[0]
    height = encryptor_matrix.shape[1]
    depth = encryptor_matrix.shape[2]
    counter = 0
    for i in range(width):
        for j in range(height):
            for k in range(depth):
                if counter >= len(password_values):
                    counter = 0
                encryptor_matrix[i][j][k] = (encryptor_matrix[i][j][k] * password_values[counter])
                counter += 1
    width = encryptor_matrix.shape[0]
    height = encryptor_matrix.shape[1]
    depth = encryptor_matrix.shape[2]
    image_pixels = np.array(image)
    for i in range(height):
        for j in range(width):
            for k in range(depth):
                encrypted_image[i][j][k] = (encryptor_matrix[i][j][k] ^ (~image_pixels[i][j][k]))

    return encrypted_image

# Now, you can call the encryptor function with the proper image path and password
row_count = config.shape[0]
for i in range(1, row_count):
    image_path = config.loc[i, 'path']
    image_path= image_path.replace("_", "_")
    image = 'Samples/' + image_path
    encrypted_image = encryptor(image, 'abc123')
    output_path = 'EncryptedSamples//Encrypted_'+image_path
    cv2.imwrite(output_path, encrypted_image)
    celebrities = amazon_celebrity_rekognition_encryption(output_path)


# In[117]:


config.head(50)


# In[119]:


import pandas as pd

# Replace 'column_name' with the actual name of the column you want to analyze
column_name = 'match_conf_original'

# Calculate the total number of values in the column
total_values = config[column_name].count()

# Calculate the number of values greater than 50
values_greater_than_50 = config[config[column_name] > 50][column_name].count()

# Compute the percentage of values greater than 50
percentage_greater_than_50 = (values_greater_than_50 / total_values) * 100

print(f"the recognition rate without encryption is: {percentage_greater_than_50:.2f}%")


# In[120]:


import pandas as pd

# Replace 'column_name' with the actual name of the column you want to analyze
column_name = 'match_conf_encryption'

# Calculate the total number of values in the column
total_values = config[column_name].count()

# Calculate the number of values greater than 50
values_less_than_50 = config[config[column_name] < 50][column_name].count()

# Compute the percentage of values greater than 50
percentage_less_than_50 = (values_less_than_50 / total_values) * 100

print(f"the recognition rate without encryption is: {percentage_less_than_50:.2f}%")


# In[122]:


result_cofig = 'config.csv'
get_ipython().run_line_magic('rm', '-rf $result_cofig')
config.to_csv(result_cofig, index=False)


# In[ ]:




