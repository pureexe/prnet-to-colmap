import os
import numpy as np
from utils.database import COLMAPDatabase,blob_to_array

DATABASE_PATH = 'sqlite/putin_68kpts.db'
DIRECTORY = 'keypoints'
POSTFIX = '_kpt.txt'
IMAGE_EXTENSION = '.jpg'
if os.path.exists(DATABASE_PATH):
    os.remove(DATABASE_PATH)
db = COLMAPDatabase.connect(DATABASE_PATH)
db.create_tables()

data_files = [f for f in os.listdir(DIRECTORY) if f[-len(POSTFIX):] == POSTFIX]

#add camera
cam_param = np.asarray([307.2, 128.0, 128.0, 0.0])
for i in range(len(data_files)):
    cam_id = db.add_camera(model=2, width=256, height=256, params=cam_param)
    file_name = data_files[i].split(POSTFIX)[0] + IMAGE_EXTENSION
    db.add_image(file_name,cam_id)

# build keypoint (feature extraction)
kpt_len = 0
for image_id, data_file in enumerate(data_files,start=1):
    keypoints = np.loadtxt(os.path.join(DIRECTORY,data_file))
    keypoints = keypoints[:,:2]
    kpt_len = keypoints.shape[0]
    db.add_keypoints(image_id, np.asarray(keypoints))

# feature mathcing
image_matchs = list()
for from_id in range(1,len(data_files)+1):
    for to_id in range(from_id+1,len(data_files)+1):
        nrange = np.arange(0,kpt_len)
        pairs = np.c_[nrange,nrange]
        # add match to database
        #db.add_matches(from_id, to_id, np.asarray(pairs))
        db.add_two_view_geometry(from_id, to_id, pairs)

# commit (save) to db 
db.commit()
db.close()
