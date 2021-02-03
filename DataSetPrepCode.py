#dataset prep code

'''
Link for data - https://www.kaggle.com/c/dog-breed-identification
train dir contains 10K dogs images
image name is id of image
labels.csv contains id vs breed mapping
'''

import pandas as pd
import os
from shutil import copyfile


base_dir='C:/Users/Sreehasa/Desktop/Miniproj/A-DogBreed/DataSet-DogBreed/'

labels = pd.read_csv(base_dir+'labels.csv')

breeds=["golden_retriever", "dingo", "pug"]

for i,row in labels.iterrows():
    print(i,row)
    print(row['breed'])
    
    if (row['breed']) in breeds:
        try:
            copyfile(base_dir+"/train/"+row['id']+".jpg", base_dir+'classes/'+row['breed']+"/"+row['id']+".jpg")
            
        except:
            if not os.path.exists(base_dir+row['breed']):
                os.makedirs(base_dir+row['breed'])
                copyfile(base_dir+"/train/"+row['id']+".jpg", base_dir+'classes/'+row['breed']+"/"+row['id']+".jpg")
                
 