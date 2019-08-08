import os, time, pickle, argparse, network_model, utils
import torch
import numpy as np
from torch.nn import functional as F  # Function-style
from torchvision import transforms
import random
print(torch.__version__)
print("cuda is available? {}".format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = True

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(233)


parser = argparse.ArgumentParser(description="No Extra Help Information Now")
parser.add_argument('--test_data', default='test',  help='test real data fold_name in data/source')
parser.add_argument('--vgg_model', help='pre-trained VGG16 model (relative) path')
parser.add_argument('--batchsize', type=int, default=4, help='batch_size')
parser.add_argument('--pre_train_epoch', type=int, default=50, help='pretrain epochs, default=10')
parser.add_argument('--adv_train_epoch', type=int, default=50, help='train epochs, default=100')
parser.add_argument('--lrG', type=float, default=0.00003, help='G learning rate, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.00003, help='D learning rate, default=0.0002')
parser.add_argument('--con_lambda', type=float, default=0.001, help='lambda for content loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--latest_pretrain_generator_model', default='', help='latest_pretrain_generator_model')
parser.add_argument('--latest_adv_generator_model', default='', help='latest_adv_generator_model')
parser.add_argument('--latest_discriminator_model', default='', help='latest_discriminator_model')
args = parser.parse_args()
print('------------ Options -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------\n')


# create result(reconstruction and Tranfer) dirs    (using test data)
if not os.path.isdir(os.path.join('results', 'only_G')):
    os.makedirs(os.path.join('results', 'only_G'))
only_G_result_path = os.path.join('results', 'only_G')
if not os.path.isdir(os.path.join('results', 'G_with_D')):
    os.makedirs(os.path.join('results', 'G_with_D'))
G_with_D = os.path.join('results', 'G_with_D')


source_data_path = os.path.join('data', 'source')
target_data_path = os.path.join('data', 'target')


# data_loader
source_data_transform = transforms.Compose([
        transforms.Resize([400, 400]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # feature scaling 均值 方差
])
edge_data_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
])

train_loader_source = utils.data_load(source_data_path, 'train1',  source_data_transform, args.batchsize, shuffle=False, drop_last=True)
train_loader_edge = utils.data_load(target_data_path, 'edge1',  edge_data_transform, args.batchsize, shuffle=False, drop_last=True)
train_loader_edge_notc = utils.data_load(target_data_path, 'edge1_not_closed', edge_data_transform, args.batchsize, shuffle=False, drop_last=True)
test_loader_source = utils.data_load(source_data_path, args.test_data, source_data_transform, 1, shuffle=False, drop_last=True)
print("train_source:  batch_num {}    batch_size {} total_samples_num {}".format(len(train_loader_source),
                                                                             train_loader_source.batch_size, len(train_loader_source.dataset)))
print("train_edge:  batch_num {} batch_size {} total_samples_num {}".format(len(train_loader_edge),
                                                                               train_loader_edge.batch_size, len(train_loader_edge.dataset)))
print("train_edge_notclosed:  batch_num {} batch_size {} total_samples_num {}".format(len(train_loader_edge_notc),
                                                                               train_loader_edge_notc.batch_size, len(train_loader_edge_notc.dataset)))
print("test_source:  batch_num {}     batch_size {} total_samples_num {}\n".format(len(test_loader_source),
                                                                           test_loader_source.batch_size, len(test_loader_source.dataset)))





# networks
G = network_model.Generator(400, 400)
G.update_pre_train_para_with_VGG16()
latest_G_path = args.latest_adv_generator_model
if latest_G_path == '':
    latest_G_path = args.latest_pretrain_generator_model
if latest_G_path != '':
    if torch.cuda.is_available():
        G.load_state_dict(torch.load(latest_G_path))
    else:
        G.load_state_dict(torch.load(latest_G_path, map_location=lambda storage, loc: storage))


D = network_model.Discriminator()
if args.latest_discriminator_model != '':
    if torch.cuda.is_available():
        D.load_state_dict(torch.load(args.latest_discriminator_model))
    else:
        D.load_state_dict(torch.load(args.latest_discriminator_model, map_location=lambda storage, loc: storage))


'''
    optimizer
'''
mean_filters = torch.ones(1, 1, 3, 3) / 9.0
mean_filters = mean_filters.to(device)

def pre_train():
    pre_train_hist = dict()
    pre_train_hist['Recon_Loss'] = []
    pre_train_hist['per_epoch_time'] = []
    pre_train_hist['total_time'] = []
    G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2), weight_decay=2e-5)
    G_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=G_optimizer, milestones=[args.pre_train_epoch // 2, args.pre_train_epoch // 4 * 3], gamma=0.1)

    print('Pre-training start!')
    start_time = time.time()
    for epoch in range(args.pre_train_epoch):
        epoch_start_time = time.time()
        Recon_epoch_losses = []
        G_scheduler.step()
        for (x, _), (y, _) in zip(train_loader_source, train_loader_edge):
            x = x.to(device)
            y = (y > 0.00001).float().detach()  # thinner edge
            y = y.to(device)  # bs*1*h*w
            # print(y[0,0,250:280,250:280])
            G_optimizer.zero_grad()
            x_out = G(x)
            weight = torch.sum(y) / torch.tensor(args.batchsize * 400 * 400)
            weight1 = torch.ones_like(y) * weight
            weight2 = torch.ones_like(y) * (1-weight)
            # 对y做一个3*3卷积 大于0的地方都加重权 这样label为non edge的也会加权
            out = F.conv2d(y, mean_filters, stride=1, padding=1)
            BCE_loss = torch.nn.BCELoss(weight=torch.where(out > 0.00001, weight2, weight1)).to(device)
            Recon_loss = BCE_loss(x_out, y)
            Recon_epoch_losses.append(Recon_loss)
            pre_train_hist['Recon_Loss'].append(Recon_loss.item())

            Recon_loss.backward()
            G_optimizer.step()

        per_epoch_time = time.time() - epoch_start_time
        pre_train_hist['per_epoch_time'].append(per_epoch_time)
        print('[{}/{}] - time: {:.2f}s,  Recon Loss: {:.3f}'.format(epoch+1, args.pre_train_epoch, per_epoch_time, torch.mean(torch.tensor(Recon_epoch_losses, dtype=torch.float32)).item()))


        if epoch % 3 == 0 or epoch == args.pre_train_epoch-1:
            with torch.no_grad():
                G.eval()
                test(G, epoch, 4, only_G_result_path)
                if epoch == args.pre_train_epoch-1:
                    torch.save(G.state_dict(), os.path.join('results', 'generator_pretrain.pkl'))

    pre_train_hist['total_time'].append(time.time() - start_time)
    with open(os.path.join('results', 'pre_train_hist.pkl'), 'wb') as f:
        pickle.dump(pre_train_hist, f)


def adv_train():
    train_hist = {}
    train_hist['Disc_loss'] = []
    train_hist['Gen_loss'] = []
    train_hist['Con_loss'] = []
    train_hist['per_epoch_time'] = []
    train_hist['total_time'] = []
    G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2), weight_decay=2e-5)
    G_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=G_optimizer, milestones=[args.adv_train_epoch // 2, args.adv_train_epoch // 4 * 3], gamma=0.1)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2), weight_decay=2e-5)
    D_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=D_optimizer, milestones=[args.adv_train_epoch // 2, args.adv_train_epoch // 4 * 3], gamma=0.1)

    print('adversarial training start!')
    start_time = time.time()
    real = torch.ones(args.batchsize, 1).to(device)
    fake = torch.zeros(args.batchsize, 1).to(device)
    for epoch in range(args.adv_train_epoch):
        epoch_start_time = time.time()
        G.train()
        G_scheduler.step()
        D_scheduler.step()
        Gen_epoch_losses = []
        Disc_epoch_losses = []
        Con_epoch_losses = []
        for (x, _), (y, _) in zip(train_loader_source, train_loader_edge):
            x = x.to(device)
            y = (y > 0.00001).float()
            y = y.to(device)

            # train D
            D_optimizer.zero_grad()

            D_real = D(y)
            BCE_loss = torch.nn.BCELoss().to(device)
            D_real_loss = BCE_loss(D_real, real)

            x_out = G(x)
            x_out = x_out.round()  # x_out变为二值  可能不能求导?
            D_fake = D(x_out)
            D_fake_loss = BCE_loss(D_fake, fake)

            Disc_loss = D_real_loss + D_fake_loss
            Disc_epoch_losses.append(Disc_loss.item())
            train_hist['Disc_loss'].append(Disc_loss.item())

            Disc_loss.backward()
            D_optimizer.step()


            # train G
            G_optimizer.zero_grad()
            x_out = G(x)
            D_fake = D(x_out)
            D_fake_loss = BCE_loss(D_fake, real)

            weight = torch.sum(y) / torch.tensor(args.batchsize * 400 * 400)
            weight1 = torch.ones_like(y) * weight
            weight2 = torch.ones_like(y) * (1-weight)
            # 对y做一个3*3卷积 大于0的地方都加重权 这样label为non edge的也会加权
            out = F.conv2d(y, mean_filters, stride=1, padding=1)
            BCE_loss = torch.nn.BCELoss(weight=torch.where(out > 0.00001, weight2, weight1)).to(device)
            Con_loss = BCE_loss(x_out, y.detach())


            Gen_loss = args.con_lambda * D_fake_loss + Con_loss
            Gen_epoch_losses.append(D_fake_loss.item())
            train_hist['Gen_loss'].append(D_fake_loss.item())
            Con_epoch_losses.append(Con_loss.item())
            train_hist['Con_loss'].append(Con_loss.item())

            Gen_loss.backward()
            G_optimizer.step()

        per_epoch_time = time.time() - epoch_start_time
        train_hist['per_epoch_time'].append(per_epoch_time)
        print('[{}/{}] - time: {:.2f}s,  Disc Loss: {:.3f},  Gen Loss: {:.3f},  Con Loss: {:.3f}'.format(epoch + 1, args.adv_train_epoch, per_epoch_time,
                                                                    torch.mean(torch.tensor(Disc_epoch_losses, dtype=torch.float32)).item(),
                                                                    torch.mean(torch.tensor(Gen_epoch_losses, dtype=torch.float32)).item(),
                                                                    torch.mean(torch.tensor(Con_epoch_losses, dtype=torch.float32)).item()))
        if epoch % 3 == 0 or epoch == args.adv_train_epoch - 1:
            with torch.no_grad():
                G.eval()
                test(G, epoch, 4, G_with_D)
                if epoch == args.adv_train_epoch - 1:
                    torch.save(G.state_dict(), os.path.join('results', 'generator_advtrain.pkl'))
                    torch.save(D.state_dict(), os.path.join('results', 'discriminator_latest.pkl'))

    total_time = time.time() - start_time
    train_hist['total_time'].append(total_time)
    with open(os.path.join('results', 'train_hist.pkl'), 'wb') as f:
        pickle.dump(train_hist, f)
    print("Training finish!")


def train():
    G.to(device)
    D.to(device)
    G.train()
    D.train()
    print('---------- Networks -------------')
    utils.print_network(G, framework_flag=False)
    utils.print_network(D, framework_flag=False)
    print('-----------------------------------------------')

    # pre_train
    pre_train()

    # adv_train
    adv_train()


def test(G, epoch, num = 4, root_path = only_G_result_path):
    for Id, (x, _) in enumerate(train_loader_source):
        x = x.to(device)
        x_recon = G(x)
        utils.save_image(x[0], x_recon[0], os.path.join(root_path, 'train_' + str(Id + 1) + '_' + 'epoch_' + str(epoch) + '.png'))
        if Id == num:
            break

    for Id, (x, _) in enumerate(test_loader_source):
        x = x.to(device)
        x_recon = G(x)
        utils.save_image(x[0], x_recon[0], os.path.join(root_path, 'test_' + str(Id + 1) + '_' + 'epoch_' + str(epoch) + '.png'))
        if Id == num:
            break


def main():
    if latest_G_path != '':
        test(G, args.pre_train_epoch + args.adv_train_epoch)
    else:
        train()


if __name__ == '__main__':
    main()