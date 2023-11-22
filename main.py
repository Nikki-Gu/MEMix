
import torch
import torch.utils.data as data
from utils import *
import time
import numpy as np
torch.set_printoptions(precision=3, edgeitems=14, linewidth=350)
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from timm.loss import *
from model import *
from casmeii import *
from sklearn.metrics.pairwise import cosine_similarity
sys.stdout = Logger("logs/your_log.log")

def run_training(cut_alpha=1.0,mix_alpha=1.0,num_classes=5):
    #writer = SummaryWriter('./tensor_log')
    args = parse_args()
    ##data normalization for both training set
    criterion = SoftTargetCrossEntropy()#torch.nn.CrossEntropyLoss()
    criterion_val=torch.nn.CrossEntropyLoss()

    data_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(4),
        transforms.RandomCrop(224, padding=4),

    ])
    # three classes
    # mean：[0.303,0.366,0.509]
    # std:[0.119,0.135,0.187]
    # mean;[0.016,0.003]
    # std:[0.994,0.919]
    # five classes
    # mean:[0.296,0.358,0.500]
    # std:[0.123,0.136,0.185]
    # mean:[0.006,0.066]
    # std:[0.956,0.835]

    flow_normal = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.006, 0.066],
                             std=[0.956, 0.833]),
    ])
    onset_normal = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.296, 0.358, 0.500],
                             std=[0.123, 0.136, 0.185]),
    ])


    # leave one subject out protocal
    LOSO = ['17', '26', '16', '9', '5', '24', '2', '13', '4', '23', '11', '12', '8', '14', '3',
            '19', '1','18',
            '10','20', '21', '22', '15', '6', '25', '7']


    val_now = 0
    num_sum = 0
    pos_pred_ALL = torch.zeros(5)
    pos_label_ALL = torch.zeros(5)
    TP_ALL = torch.zeros(5)

    acc_list = []
    epoch_list = []

    norm_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])


    for subj in LOSO:
        train_dataset = RafDataSet(args.raf_path, phase='train', num_loso=subj,  transform_flow=flow_normal,transform_onset=onset_normal,transform_aug=data_aug,num_classes=num_classes)
        val_dataset = RafDataSet(args.raf_path, phase='test', num_loso=subj,  transform_flow=flow_normal,transform_onset=onset_normal,num_classes=num_classes)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=34,
                                                   num_workers=args.workers,
                                                   shuffle=True,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=34,
                                                 num_workers=args.workers,
                                                 shuffle=False,
                                                 pin_memory=True)
        print('num_sub', subj)
        print('Train set size:', train_dataset.__len__())
        print('Validation set size:', val_dataset.__len__())

        max_epoch = 0
        max_corr = 0
        max_f1 = 0
        max_pos_pred = torch.zeros(5)
        max_pos_label = torch.zeros(5)
        max_TP = torch.zeros(5)
        net_all = MixMe_Net(pretrained=True, num_classes=num_classes, return_attn=True,merge=True)

        optimizer_all = torch.optim.Adam(net_all.parameters(),lr=3e-5)


        net_all = net_all.cuda()

        for i in range(1, 100):
            running_loss = 0.0
            correct_sum = 0
            iter_cnt = 0

            net_all.train()

            for batch_i, (
                    flow,onset, label_all) in enumerate(train_loader):
                iter_cnt += 1
                label_all=label_all.cuda()
                onset=onset.cuda()
                flow = flow.cuda()
                mixup_lam = np.random.beta(mix_alpha, mix_alpha)
                mask,lam=generate_flow_mask(flow,lam_alpha=cut_alpha,type='avg')
                mask_224=torch.nn.functional.interpolate(mask,size=(224,224),mode='nearest')
                index = torch.randperm(flow.size()[0]).cuda()
                label1=label_all
                label2=label_all[index]
                mix_flow = flow*(mask_224) + mixup_lam * flow * (1 - mask_224) + (1 - mixup_lam) * flow[index] * (
                            1 - mask_224)
                ALL, attn, final = net_all(mix_flow, onset)
                label_mix=label_mix = transmix_label(label1, label2, mask, attn, mixup_lam, num_classes=num_classes,
                                           smoothing=0.0)

                loss_all = criterion(ALL,label_mix)#loss_func(criterion,ALL_mix)
                #writer.add_scalar('trans_loss_mixup/'+subj, loss_all, i)
                optimizer_all.zero_grad()
                loss_all.backward()
                optimizer_all.step()


                running_loss += loss_all
                _, predicts = torch.max(ALL, 1)
                correct_num = torch.eq(predicts, label_all).sum()
                correct_sum += correct_num

            if i >= 0:
                acc = correct_sum.float() / float(train_dataset.__len__())

                running_loss = running_loss / iter_cnt

                print('[Epoch %d] Training Loss: %.3f' % (i, running_loss))

            pos_label = torch.zeros(5)
            pos_pred = torch.zeros(5)
            TP = torch.zeros(5)

            with torch.no_grad():
                running_loss = 0.0
                iter_cnt = 0
                bingo_cnt = 0
                sample_cnt = 0
                net_all.eval()
                for batch_i, ( flow,onset,label_all) in enumerate(val_loader):
                    label_all = label_all.cuda()
                    flow = flow.cuda()
                    onset=onset.cuda()
                    ##test
                    (ALL,attn,_) = net_all(flow,onset)
                    loss = criterion_val(ALL, label_all)
                    running_loss += loss
                    iter_cnt += 1
                    _, predicts = torch.max(ALL, 1)
                    correct_num = torch.eq(predicts, label_all)
                    bingo_cnt += correct_num.sum().cpu()
                    sample_cnt += ALL.size(0)

                    for cls in range(5):

                        for element in predicts:
                            if element == cls:
                                pos_label[cls] = pos_label[cls] + 1
                        for element in label_all:
                            if element == cls:
                                pos_pred[cls] = pos_pred[cls] + 1
                        for elementp, elementl in zip(predicts, label_all):
                            if elementp == elementl and elementp == cls:
                                TP[cls] = TP[cls] + 1
                    count = 0
                    SUM_F1 = 0
                    for index in range(5):
                        if pos_label[index] != 0 or pos_pred[index] != 0:
                            count = count + 1
                            SUM_F1 = SUM_F1 + 2 * TP[index] / (pos_pred[index] + pos_label[index])

                    AVG_F1 = SUM_F1 / count

                running_loss = running_loss / iter_cnt
                acc = bingo_cnt.float() / float(sample_cnt)
                acc = np.around(acc.numpy(), 4)
                #writer.add_scalar('trans_acc_mixup/' + subj, acc, i)
                if bingo_cnt > max_corr:
                    max_corr = bingo_cnt
                    max_epoch = i
                if AVG_F1 >= max_f1:
                    max_f1 = AVG_F1
                    max_pos_label = pos_label
                    max_pos_pred = pos_pred
                    max_TP = TP
                print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f, F1-score:%.3f" % (i, acc, running_loss, AVG_F1))
                if acc==1.:
                    print('achieve 100%acc, break')
                    break
        num_sum = num_sum + max_corr
        pos_label_ALL = pos_label_ALL + max_pos_label
        pos_pred_ALL = pos_pred_ALL + max_pos_pred
        TP_ALL = TP_ALL + max_TP
        count = 0
        SUM_F1 = 0
        for index in range(5):
            if pos_label_ALL[index] != 0 or pos_pred_ALL[index] != 0:
                count = count + 1
                SUM_F1 = SUM_F1 + 2 * TP_ALL[index] / (pos_pred_ALL[index] + pos_label_ALL[index])

        F1_ALL = SUM_F1 / count
        val_now = val_now + val_dataset.__len__()
        acc_now=(max_corr/val_dataset.__len__()).item()
        #writer.add_scalar('trans_acc', acc_now, subj)
        print("[subject %s] correct_num:%d sum:%d ACC: %.4f  " % (subj, max_corr, val_dataset.__len__(),acc_now))
        print("[ALL_corr]: %d [ALL_val]: %d [ALL_ACC]:%.4f" % (int(num_sum), int(val_now), num_sum / val_now))
        print("[F1_now]: %.4f [F1_ALL]: %.4f" % (max_f1, F1_ALL))
        print('max_epoch:', str(max_epoch))
        acc_list.append(round(acc_now,4))
        epoch_list.append(str(max_epoch))
    print("--------------------------------------------------------------------------------------------------")
    print('cut alpha,', cut_alpha)
    print('mix alpha', mix_alpha)
    print("Acc_list:", acc_list)
    print('max epoches:', epoch_list)
    print("Total acc: %.4f"% (num_sum * 1.0 / val_now))

if __name__ == "__main__":
    seed=2
    start_time = time.time()
    seed_torch(seed)
    run_training(cut_alpha=2.0, mix_alpha=1.0,num_classes=5)
    end_time = time.time()
    print("start time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    print("end_time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))


