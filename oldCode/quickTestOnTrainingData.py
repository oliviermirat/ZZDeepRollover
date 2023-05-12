result = model.predict(tot)
result.shape
np.argmax(result, axis=-1)

np.sum(np.argmax(result[0:1617], axis=-1))
np.sum(np.argmax(result[1617:3234], axis=-1))

directory_path = data_root + '/rollover/'
tot1 = []
directory = os.fsencode(directory_path)
for file in os.listdir(directory):
  name = os.fsdecode(file)
  if name.endswith(".png"): 
    filename = directory_path + name
    im = cv2.imread(filename)
    im1 = cv2.resize(im,(224,224))
    tot1.append(im1)
directory_path = data_root + '/normal/'
tot2 = []
directory = os.fsencode(directory_path)
for file in os.listdir(directory):
  name = os.fsdecode(file)
  if name.endswith(".png"): 
    filename = directory_path + name
    im = cv2.imread(filename)
    im1 = cv2.resize(im,(224,224))
    tot2.append(im1)

result = model.predict(np.array(tot1))
result.shape
np.argmax(result, axis=-1)
np.sum(np.argmax(result, axis=-1))

#####
#####
#####

test_root = 'testSet'
directory_path = test_root + '/rollover/'
maxiter = 200
iter = 0
tot1 = []
directory = os.fsencode(directory_path)
for file in os.listdir(directory):
  name = os.fsdecode(file)
  if name.endswith(".png"): 
    filename = directory_path + name
    im = cv2.imread(filename)
    im1 = cv2.resize(im,(224,224))
    im1 = im1 / 255
    if iter < maxiter:
      tot1.append(im1)
      iter = iter + 1
    else:
      break
directory_path = test_root + '/normal/'
iter = 0
tot2 = []
directory = os.fsencode(directory_path)
for file in os.listdir(directory):
  name = os.fsdecode(file)
  if name.endswith(".png"): 
    filename = directory_path + name
    im = cv2.imread(filename)
    im1 = cv2.resize(im,(224,224))
    im1 = im1 / 255
    if iter < maxiter:
      tot2.append(im1)
      iter = iter + 1
    else:
      break
# tot = [tot1, tot2]
tot = np.concatenate((tot1, tot2), axis=0)

result = model.predict(tot)
result.shape
np.argmax(result, axis=-1)

np.sum(np.argmax(result[0:200], axis=-1))
np.sum(np.argmax(result[200:400], axis=-1))