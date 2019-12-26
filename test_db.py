from utils.database import COLMAPDatabase,blob_to_array,pair_id_to_image_ids
import numpy as np
import json

db = COLMAPDatabase.connect('sqlite/putin_ori.db')
c = db.cursor()
c.execute("SELECT data From two_view_geometries LIMIT 1 OFFSET 5")
data = c.fetchall()
print(data)
data = blob_to_array(data[0][0],np.int32)
print(data.reshape(44,2))
db.commit()
db.close()