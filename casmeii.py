import torch.utils.data as data
import pandas as pd
import os, torch
import cv2
from torchvision import transforms

class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, num_loso, transform_flow=None,transform_onset=None,transform_aug=None,num_classes=5):
        self.phase = phase
        self.raf_path = raf_path
        self.transform_flow = transform_flow
        self.transform_onset=transform_onset
        self.transform_aug=transform_aug
        SUBJECT_COLUMN = 0
        NAME_COLUMN = 1
        ONSET_COLUMN = 2
        APEX_COLUMN = 3
        OFF_COLUMN = 4
        LABEL_AU_COLUMN = 5
        LABEL_ALL_COLUMN = 6

        df = pd.read_excel(os.path.join(self.raf_path, 'CASME2-coding-20140508.xlsx'), usecols=[0, 1, 3, 4, 5, 7, 8])
        df['Subject'] = df['Subject'].apply(str)

        if phase == 'train':
            dataset = df.loc[df['Subject'] != num_loso]
        else:
            dataset = df.loc[df['Subject'] == num_loso]

        Subject = dataset.iloc[:, SUBJECT_COLUMN].values
        File_names = dataset.iloc[:, NAME_COLUMN].values
        Label_all = dataset.iloc[:,
                    LABEL_ALL_COLUMN].values  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        Onset_num = dataset.iloc[:, ONSET_COLUMN].values
        Apex_num = dataset.iloc[:, APEX_COLUMN].values
        Offset_num = dataset.iloc[:, OFF_COLUMN].values
        Label_au = dataset.iloc[:, LABEL_AU_COLUMN].values
        self.file_paths_on = []
        self.file_paths_apex = []
        self.file_paths_off=[]
        self.file_names = []
        self.label= []
        self.sub = []

        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        # use aligned images for training/testing
        for (f, sub, onset, apex, offset, label_all, label_au) in zip(File_names, Subject, Onset_num, Apex_num,
                                                                      Offset_num, Label_all, Label_au):
            #three classes
            if num_classes == 3:
                if label_all == 'happiness' or label_all == 'repression' or label_all == 'disgust' or label_all == 'surprise' or label_all == 'fear' or label_all == 'sadness' :#or label_all=='others':

                    self.file_paths_on.append(onset)
                    self.file_paths_off.append(offset)
                    self.file_paths_apex.append(apex)
                    self.sub.append(sub)
                    self.file_names.append(f)
                    if label_all == 'happiness':
                        self.label.append(0)
                        a=a+1
                    elif label_all == 'surprise':
                        self.label.append(1)
                        b=b+1
                    else:
                        self.label.append(2)
                        c=c+1
            elif num_classes==5:
            #five classes
                if label_all == 'happiness' or label_all == 'repression' or label_all == 'disgust' or label_all == 'surprise' or label_all == 'others':

                    self.file_paths_on.append(onset)
                    self.file_paths_off.append(offset)
                    self.file_paths_apex.append(apex)
                    self.sub.append(sub)
                    self.file_names.append(f)

                    if label_all == 'happiness':
                        self.label.append(0)

                        a = a + 1
                    elif label_all == 'repression':
                        self.label.append(1)

                        b = b + 1
                    elif label_all == 'disgust':
                        self.label.append(2)

                        c = c + 1
                    elif label_all == 'surprise':
                        self.label.append(3)

                        d = d + 1
                    else:
                        self.label.append(4)
                        e = e + 1
            else:
                print('wrong')

    def __len__(self):
        return len(self.label)


    def __getitem__(self, idx):
        ##sampling strategy for training set

        onset = self.file_paths_on[idx]
        apex = self.file_paths_apex[idx]
        off = self.file_paths_off[idx]
        on0=str(onset)
        apex0=str(apex)
        #off0=str(off)
        sub = str(self.sub[idx])
        f = str(self.file_names[idx])

        on0_j = 'reg_img' + on0 + '.jpg'
        apex0_j = 'reg_img' + apex0 + '.jpg'
        #off0_j='reg_img'+ off0 +'.jpg'
        path_on0 = os.path.join(self.raf_path, 'Cropped-updated', 'Cropped', 'sub%02d' % int(sub), f, on0_j)
        path_apex0 = os.path.join(self.raf_path, 'Cropped-updated', 'Cropped', 'sub%02d' % int(sub), f, apex0_j)
       # path_off0 = os.path.join(self.raf_path, 'Cropped-updated', 'Cropped', 'sub%02d' % int(sub), f, off0_j)
        image_on0 = cv2.imread(path_on0)
        image_apex0 = cv2.imread(path_apex0)
        image_apex0 = cv2.resize(image_apex0, (224, 224))
        image_on0 = cv2.resize(image_on0, (224, 224))
        if self.phase!='train':
            image_on = cv2.cvtColor(image_on0, cv2.COLOR_BGR2GRAY)
            image_apex = cv2.cvtColor(image_apex0, cv2.COLOR_BGR2GRAY)
            flow_1 = cv2.calcOpticalFlowFarneback(image_on, image_apex, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            label = self.label[idx]
            flow_1 = self.transform_flow(flow_1)
            image_on0 = self.transform_onset(image_on0)
            if self.transform_aug is not None:
                flow_1 = self.transform_aug(flow_1)
                image_on0 = self.transform_aug(image_on0)
            return flow_1, image_on0, label
        else:
            label= self.label[idx]
            #flow_1=self.transform_flow(flow_1)
            image_on0=self.transform_onset(image_on0)
            image_apex0 = self.transform_onset(image_apex0)
            if self.transform_aug is not None:
                #flow_1 = self.transform_aug(flow_1)
                image_on0 = self.transform_aug(image_on0)
                image_apex0 = self.transform_aug(image_apex0)
            return image_on0,image_apex0,label
