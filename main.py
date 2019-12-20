from utils.database import COLMAPDatabase,blob_to_array
import numpy as np
import json

db = COLMAPDatabase.connect('sqlite/hall.db')
c = db.cursor()
c.execute("SELECT data FROM matches LIMIT 1")
data = c.fetchall()
matches_data = blob_to_array(data[0][0],np.uint32)
np.savetxt('data.txt',matches_data)