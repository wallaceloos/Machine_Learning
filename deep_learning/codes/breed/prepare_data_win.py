import pandas as pd
import numpy as np
import os
import sys
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


DATASET_DIR = " "#path of the dataset
TRAIN_DIR = DATASET_DIR+"\\train"
TRAIN_LABEL = DATASET_DIR+"\\labels.csv"
DATASET_MULTICLASS_DIR = DATASET_DIR+"\\dataset_multiclass"
TRAIN_SET_DIR = DATASET_DIR+"\\train_set"
VAL_SET_DIR = DATASET_DIR+"\\val_set"
TEST_SET_DIR = DATASET_DIR+"\\test_set"
	

def createFoldersClasses():

	df_train = pd.read_csv(TRAIN_LABEL)
	breeds = df_train.breed.unique()
	breeds = np.sort(breeds)

	if not os.path.exists(DATASET_DIR+"\\dataset_multiclass"):
		  os.makedirs(DATASET_DIR+"\\dataset_multiclass")
		  
	for name in tqdm(breeds):
		  if not os.path.exists(DATASET_DIR+"\\dataset_multiclass\\"+name):
		      os.makedirs(DATASET_DIR+"\\dataset_multiclass\\"+name)

def copyToFolders():
	df_train = pd.read_csv(TRAIN_LABEL)
	for index, row in tqdm(df_train.iterrows()):
		  src = TRAIN_DIR+"\\"+row['id']+".jpg"
		  dst = DATASET_MULTICLASS_DIR+"\\"+row['breed']
		  shutil.copy (src, dst)

def createTrainTestValFolders():
	if not os.path.exists(DATASET_DIR+"\\train_set"):
		  os.makedirs(DATASET_DIR+"\\train_set")
		  
	if not os.path.exists(DATASET_DIR+"\\val_set"):
		  os.makedirs(DATASET_DIR+"\\val_set")
		  
	if not os.path.exists(DATASET_DIR+"\\test_set"):
		  os.makedirs(DATASET_DIR+"\\test_set")

def moveImgsFolders():
	train_len =  0 
	test_len =  0 
	val_len =  0 

	print("Creating Training/Validation/Test Set")
	for filename in tqdm(os.listdir(DATASET_MULTICLASS_DIR)):
		  files = os.listdir(DATASET_MULTICLASS_DIR+'\\'+filename)
		  files = np.asarray(files)
		  
		  if not os.path.exists(TRAIN_SET_DIR+"\\"+filename):
		      os.makedirs(TRAIN_SET_DIR+"\\"+filename)
		      
		  if not os.path.exists(VAL_SET_DIR+"\\"+filename):
		      os.makedirs(VAL_SET_DIR+"\\"+filename)
		  
		  if not os.path.exists(TEST_SET_DIR+"\\"+filename):
		      os.makedirs(TEST_SET_DIR+"\\"+filename)
		  
		  train_sz = int(0.8 * files.shape[0])
		  test_sz = files.shape[0] - train_sz
		  val_sz = int(0.2 * train_sz)
		  train_sz = train_sz-val_sz
		  
		  train_files = files[0:train_sz]
		  val_files = files[train_sz:train_sz+val_sz]
		  test_files = files[train_sz+val_sz:train_sz+val_sz+test_sz]
		
		  for img_name in train_files:
		      img_path = DATASET_MULTICLASS_DIR+'\\'+filename+'\\'+img_name
		      dst = TRAIN_SET_DIR+"\\"+filename
		      shutil.move(img_path,dst)

		  for img_name in val_files:
		      img_path = DATASET_MULTICLASS_DIR+'\\'+filename+'\\'+img_name
		      dst = VAL_SET_DIR+"\\"+filename
		      shutil.move(img_path,dst)
		  
		  for img_name in test_files:
		      img_path = DATASET_MULTICLASS_DIR+'\\'+filename+'\\'+img_name
		      dst = TEST_SET_DIR+"\\"+filename
		      shutil.move(img_path,dst)
		  
		  train_len = train_len + train_sz
		  val_len = val_len + val_sz
		  test_len = test_len + test_sz

if __name__ == '__main__':
	
	createFoldersClasses()
	copyToFolders()
	createTrainTestValFolders()
	moveImgsFolders()




