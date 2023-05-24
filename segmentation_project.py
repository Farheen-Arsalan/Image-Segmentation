import matplotlib.image as mimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
path = '/Users/Arsalan/Desktop/image.jpeg'
im = mimg.imread(path)
im_orig = im.copy()
print(im.shape)

#Analyse the data
print('min1 :' ,im[:,:,0].min())
print('min2 :' ,im[:,:,1].min())
print('min3 :' ,im[:,:,2].min())

print('max1 :' ,im[:,:,0].max())
print('max2 :' ,im[:,:,1].max())
print('max3 :' ,im[:,:,2].max())

#display image
plt.figure(1)
plt.imshow(im)
plt.axis('off')


#seperate out the channels

R = im[:,:,0]
G = im[:,:,1]
B = im[:,:,2]


plt.figure(2)
plt.subplot(2,2,1)
plt.imshow(im)
plt.title('Original image')
plt.axis('off')
plt.subplot(2,2,2)
plt.imshow(R,cmap='gray')
plt.title('R componenet image')
plt.axis('off')
plt.subplot(2,2,3)
plt.imshow(G,cmap='gray')
plt.title('G componenet image')
plt.axis('off')
plt.subplot(2,2,4)
plt.imshow(B,cmap='gray')
plt.title('B componenet image')
plt.axis('off')

#3-D Channel seperation
Rtemp = np.zeros((im.shape[0],im.shape[1],3),dtype='uint8')
Rtemp[:,:,0]=R

Gtemp = np.zeros((im.shape[0],im.shape[1],3),dtype='uint8')
Gtemp[:,:,1]=G

Btemp = np.zeros((im.shape[0],im.shape[1],3),dtype='uint8')
Btemp[:,:,2]=B

plt.figure(3)
plt.subplot(2,2,1)
plt.imshow(im)
plt.title('Original image')
plt.axis('off')
plt.subplot(2,2,2)
plt.imshow(Rtemp,cmap='gray')
plt.title('R componenet image')
plt.axis('off')
plt.subplot(2,2,3)
plt.imshow(Gtemp,cmap='gray')
plt.title('G componenet image')
plt.axis('off')
plt.subplot(2,2,4)
plt.imshow(Btemp,cmap='gray')
plt.title('B componenet image')
plt.axis('off')



#lets apply kmeans on channel
#that could be any channel
#R component for kmeans
data = R.reshape(-1,1)
mdlKmeans = KMeans(n_clusters=4)
mdlKmeans = mdlKmeans.fit(data)
labels = mdlKmeans.labels_
#unique label assign to pixels
print(np.unique(labels))

label_image = labels.reshape(im.shape[0],im.shape[1])
plt.figure(1)
plt.imshow(label_image,cmap='gray')

#Seperate out the segemsnts by having
#black and white image of each segment

seg1 = np.zeros((im.shape[0],im.shape[1]))
seg2 = np.zeros((im.shape[0],im.shape[1]))
seg3 = np.zeros((im.shape[0],im.shape[1]))
seg4 = np.zeros((im.shape[0],im.shape[1]))


ind = label_image==0
seg1[ind] = 1
segPer1 = (np.count_nonzero(seg1)/(im.shape[0]*im.shape[1]))

ind1 = label_image==1
seg2[ind1] = 1
segPer2 = (np.count_nonzero(seg2)/(im.shape[0]*im.shape[1]))

ind2 = label_image==2
seg3[ind2] = 1
segPer3 = (np.count_nonzero(seg3)/(im.shape[0]*im.shape[1]))

ind3 = label_image==3
seg4[ind3] = 1
segPer4 = (np.count_nonzero(seg4)/(im.shape[0]*im.shape[1]))

plt.figure(num=10,figsize=(10,6))
plt.subplot(2,2,1)
plt.imshow(seg1,cmap='gray')
plt.title('segment 1'+str(segPer1))
plt.axis('off')
plt.subplot(2,2,2)
plt.imshow(seg2,cmap='gray')
plt.title('segment 2'+str(segPer2))
plt.axis('off')
plt.subplot(2,2,3)
plt.imshow(seg3,cmap='gray')
plt.title('segment 3'+str(segPer3))
plt.axis('off')
plt.subplot(2,2,4)
plt.imshow(seg4,cmap='gray')
plt.title('segment 4'+str(segPer4))
plt.axis('off')

per = np.array([segPer1,segPer2,segPer3,segPer4])

#where we have petal
cat = np.argmin(per)


final_segm = cat+1
if(final_segm==1):
    segmentImage = seg1
elif(final_segm==2):
    segmentImage = seg2
elif(final_segm==3):
    segmentImage = seg3

elif(final_segm==4):
    segmentImage = seg4
    
    
#convert segmented image to color
IMcolor = im.copy()
IMred = IMcolor[:,:,0]
IMgrn = IMcolor[:,:,1]
IMblu = IMcolor[:,:,2]













