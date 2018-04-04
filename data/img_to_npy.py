from PIL import Image
import os
import numpy as np
normal_file_names = os.listdir('./Normal')
abnormal_file_names = os.listdir('./Abnormal')
targets = [[0]*len(normal_file_names), [1]*len(abnormal_file_names)]
images = []

for normal_file in normal_file_names:
	image  = np.array(Image.open(os.path.join('Normal', normal_file)).convert('RGB').resize((224,224),Image.ANTIALIAS))
	
	images.append(image)
for abnormal_file in abnormal_file_names:
	image  = np.array(Image.open(os.path.join('Abnormal', abnormal_file)).convert('RGB').resize((224,224), Image.ANTIALIAS))
	images.append(image)
		
images  = np.array(images)
targets = np.concatenate(np.array(targets))
print('saving .....')
np.save('images.npy', images)
np.save('targets.npy', targets)
print(images[1].shape, targets)
