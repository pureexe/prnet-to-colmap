from utils.database import COLMAPDatabase,blob_to_array,pair_id_to_image_ids
import numpy as np
import json

db = COLMAPDatabase.connect('C:\\colmap\\datasets\\putin\\putin_twoview.db')
c = db.cursor()
c.execute("SELECT F From two_view_geometries LIMIT 1 OFFSET 50 ")
data = c.fetchall()
print(data)
data = blob_to_array(data[0][0],np.float64)
print(data.reshape(3,3))
db.commit()
db.close()