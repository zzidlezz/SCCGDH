import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import dset
from models import ImgNet, TxtNet, LabelNet
from metric import compress, calculate_map, calculate_top_map
import settings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Session:
    def __init__(self):
        self.logger = settings.logger

        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)

        self.train_dataset = dset.MY_DATASET(train=True, transform=dset.train_transform)
        self.test_dataset = dset.MY_DATASET(train=False, database=False, transform=dset.test_transform)
        self.database_dataset = dset.MY_DATASET(train=False, database=True, transform=dset.test_transform)

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=settings.BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=settings.NUM_WORKERS,
                                                   drop_last=True,
                                                    pin_memory=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=settings.BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=settings.NUM_WORKERS,
                                                       pin_memory=True
                                                       )

        self.database_loader = torch.utils.data.DataLoader(dataset=self.database_dataset,
                                                  batch_size=settings.BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=settings.NUM_WORKERS,
                                                    pin_memory=True
                                                           )

        self.CodeNet_I = ImgNet(code_len=settings.CODE_LEN)
        self.FeatNet_I = ImgNet(code_len=settings.CODE_LEN)
        self.labelnet = LabelNet(code_len=settings.CODE_LEN,label_dim=settings.LABEL_DIM)


        txt_feat_len = dset.txt_feat_len
        self.CodeNet_T = TxtNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)

        self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)

        self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=settings.LR_TXT, momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)

        self.opt_L = torch.optim.SGD(self.labelnet.parameters(), lr=settings.LR_TXT, momentum=settings.MOMENTUM,
                                     weight_decay=settings.WEIGHT_DECAY)

    def train(self, epoch):
        self.CodeNet_I.cuda().train()
        self.FeatNet_I.cuda().eval()
        self.CodeNet_T.cuda().train()
        self.labelnet.cuda().train()

        self.CodeNet_I.set_alpha(epoch)
        self.CodeNet_T.set_alpha(epoch)
        self.labelnet.set_alpha(epoch)

        self.logger.info('Epoch [%d/%d], alpha for ImgNet: %.3f, alpha for TxtNet: %.3f' % (epoch + 1, settings.NUM_EPOCH, self.CodeNet_I.alpha, self.CodeNet_T.alpha))

        for idx, (img, F_T, labels, _) in enumerate(self.train_loader):

            img = Variable(img.cuda())
            F_T = Variable(torch.FloatTensor(F_T.numpy()).cuda())
            labels = Variable(labels.cuda())

            ss_ = (labels@labels.t() > 0) * 2 - 1

            self.opt_I.zero_grad()
            self.opt_T.zero_grad()
            self.opt_L.zero_grad()

            F_I , _, _ = self.FeatNet_I(img)
            _, hid_I, code_I = self.CodeNet_I(img)
            _, hid_T, code_T = self.CodeNet_T(F_T)



            _,_,label_output = self.labelnet(labels.to(torch.float32))
            B_I = F.normalize(code_I)
            B_T = F.normalize(code_T)

            loss1 = torch.mean(
                torch.square((torch.matmul(B_I, label_output.t())) - ss_))
            loss2 = torch.mean(
                torch.square((torch.matmul(B_T, label_output.t())) - ss_))
            loss3 =  torch.mean((torch.square(B_I - B_T)))

            loss =loss1+loss2+0.1*loss3

            loss.backward()
            self.opt_I.step()
            self.opt_T.step()
            self.opt_L.step()

            if (idx + 1) % (len(self.train_dataset) // settings.BATCH_SIZE / settings.EPOCH_INTERVAL) == 0:
                self.logger.info('Epoch [%d/%d], Iter [%d/%d] Loss1: %.4f Loss2: %.4f Loss3: %.4f Total Loss: %.4f'
                    % (epoch + 1, settings.NUM_EPOCH, idx + 1, len(self.train_dataset) // settings.BATCH_SIZE,
                        loss1.item(), loss2.item(), loss3.item(), loss.item()))





    def eval(self):
        self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')

        self.CodeNet_I.eval().cuda()
        self.CodeNet_T.eval().cuda()
        self.labelnet.eval().cuda()

        t1 = time.time()

        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L= compress(self.database_loader, self.test_loader, self.CodeNet_I, self.CodeNet_T, self.database_dataset, self.test_dataset)


        # MAP_I2T = calculate_map(qu_BI, re_BT, qu_L, re_L)
        # MAP_T2I = calculate_map(qu_BT, re_BI, qu_L, re_L)

        MAP_I2T_50 = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I_50 = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)



        t2 = time.time()

        print('[%d]Test time:%.2f ...test map: map(i->t): %3.4f, map(t->i): %3.4f' % (settings.CODE_LEN, (t2 - t1), MAP_I2T_50, MAP_T2I_50))



def main():
    
    sess = Session()

    if settings.EVAL == True:
        sess.load_checkpoints()
        sess.eval()

    else:

        for epoch in range(settings.NUM_EPOCH):
            # train the Model
            sess.train(epoch)
            # eval the Model
            if (epoch + 1) % settings.EVAL_INTERVAL == 0:
                sess.eval()
            # # save the model
            # if epoch + 1 == settings.NUM_EPOCH:
            #     sess.save_checkpoints(step=epoch+1)



if __name__ == '__main__':
    if not os.path.exists('result'):
        os.makedirs('result')
    if not os.path.exists('log'):
        os.makedirs('log')

    main()

