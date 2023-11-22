from mtcnn.mtcnn import MTCNN
import cv2 as cv2
import os
def crop_first_img(img_path):
    img = cv2.imread(img_path)
    faces = model.detect_faces(img)
    assert len(faces)==1
    for face in faces:
        x,y,w,h = face['box']
        #print(x,y,w,h)
        cropped = img[y:y+h,x:x+w,:]
    return cropped, [y,y+h,x,x+w]

def crop_other_img(img_path,pos):
    img = cv2.imread(img_path)
    cropped = img[pos[0]:pos[1],pos[2]:pos[3],:]
    return cropped


raw_path='./raw_dataset_path' #change to your path
save_path='./cropped_dataset_path' #change to your path
model = MTCNN()
all_images=os.listdir(raw_path)
for i in range(len(all_images)):
    image=all_images[i]
    img_path=os.path.join(raw_path,image)
    if i==0:
        cropped,pos=crop_first_img(img_path)
    else:
        cropped=crop_other_img(img_path,pos)
    save_img_path=os.path.join(save_path,image)
    cv2.imwrite(save_img_path,cropped)