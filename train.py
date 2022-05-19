from calendar import EPOCH
from contextvars import Context
import json
from pickle import FALSE
import sys
import datetime
from pytz import timezone
from turtle import color
from xml.dom import VALIDATION_ERR

import argparse
from cProfile import label
from xml.etree.ElementTree import Comment
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
from model.build_BiSeNet import BiSeNet
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils import create_mask, get_index, save_images, parameter_flops_count, colorLabel, poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from loss import DiceLoss, flatten
import torch.cuda.amp as amp
from torchvision import transforms
from PIL import Image
from dataset.cityscapes import Cityscapes
from dataset.gta import GTA
from model.discriminator import FCDiscriminator, LightDiscriminator
import torch.nn
from torch.nn import NLLLoss
import torch.nn.functional as F



#------------------------------------------------------------------------------
#------------------------ DEFAULT PARAMETERS ----------------------------------
#------------------------------------------------------------------------------
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

NUM_EPOCHS = 50
EPOCH_START_i = 0
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4

DATA_SOURCE = './data/GTA5'
DATA_LIST_PATH_SOURCE = 'train.txt'
DATA_TARGET = './data/Cityscapes/data'
DATA_LIST_PATH_TARGET = 'train.txt'
INFO_FILE_PATH = 'info.json'

INPUT_SIZE_SOURCE = '720,1280'
INPUT_SIZE_TARGET = '512,1024'
CROP_WIDTH = '1024'
CROP_HEIGHT = '512'
RANDOM_SEED = 1234

NUM_CLASSES = 19
LEARNING_RATE = 2.5e-4
WEIGHT_DECAY = 0.0005 # Bisenet : 1e-4
MOMENTUM = 0.9
POWER = 0.9
LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1                #quali lambda servono? 
LAMBDA_ADV_TARGET1 = 0.001      #prima era 0.0002 quali lambda servono? 

PRETRAINED_MODEL_PATH = None
CONTEXT_PATH = "resnet101"
OPTIMIZER = 'sgd'
LOSS = 'crossentropy'
FLOPS = False
LIGHT = True
WITH_MASK = False
SAVE_IMAGES = True
SAVE_IMAGES_STEP = 10

TENSORBOARD_LOGDIR = 'run'
CHECKPOINT_STEP = 5
VALIDATION_STEP = 15
SAVE_MODEL_PATH = None

#------------------------------------------------------------------------------------------------------
#------------------------I seguenti parametri potrebbero non servire-----------------------------------
#------------------------------------------------------------------------------------------------------

NUM_STEPS = 250000       #An epoch consists of one full cycle through the training data. 
                         #This is usually many steps. 
                         #As an example, if you have 2,000 images and use a batch size of 10 an epoch consists of:
                         #2,000 images / (10 images / step) = 200 steps.
NUM_STEPS_STOP = 150000  # early stopping

SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots/'

GAN = 'Vanilla'
IGNORE_LABEL = 255 

TARGET = 'cityscapes'
SET = 'train'

print("Import terminato")

#------------------------------------------------------------------------------------------------------
#-------------------------------------ARGUMENTS PARSING------------------------------------------------
#------------------------------------------------------------------------------------------------------
def get_arguments(params):

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs to train for')                     # -> num_steps 
    parser.add_argument('--epoch_start_i', type=int, default=EPOCH_START_i, help='Start counting epochs from this number')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Number of images in each batch')
    parser.add_argument('--iter_size', type=int, default=ITER_SIZE, help='Accumulate gradients for iter_size iteractions')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='num of workers')
    

    parser.add_argument('--data_source', type=str, default=DATA_SOURCE, help='path of training source data')
    parser.add_argument('--data_list_path_source', type=str, default=DATA_LIST_PATH_SOURCE, help='path of training labels of source data')
    parser.add_argument('--data_target', type=str, default=DATA_TARGET, help='path of training target data')
    parser.add_argument('--data_list_path_target', type=str, default=DATA_LIST_PATH_TARGET, help='path of training labels of target data')
    parser.add_argument('--info_file', type=str, default=INFO_FILE_PATH, help='path info file')


    parser.add_argument('--input_size_source', type=str, default=INPUT_SIZE_SOURCE, help='Size of input source image')
    parser.add_argument('--input_size_target', type=str, default=INPUT_SIZE_TARGET, help='Size of input target image')
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED, help='Random seed for reproducibility')


    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES, help='num of object classes (with void)')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='learning rate used for train')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY, help='Weight decay for SGD')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum for SGD')
    parser.add_argument('--power', type=float, default=POWER, help='Power for polynomial learning rate decay')
    parser.add_argument("--learning_rate_D", type=float, default=LEARNING_RATE_D, help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,help="lambda_adv for adversarial training.")

    
    parser.add_argument('--pretrained_model_path', type=str, default=PRETRAINED_MODEL_PATH, help='path to pretrained model')
    parser.add_argument('--context_path', type=str, default=CONTEXT_PATH, help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default=LOSS, help='loss function, dice or crossentropy')
    parser.add_argument('--flops', type=bool, default=FLOPS, help='Display the number of parameter and the number of flops')
    parser.add_argument('--light', type=bool, default=LIGHT, help='Perform the training with the lightweight discriminator')
    parser.add_argument('--with_mask', type=bool, default=WITH_MASK, help='Indicate if mask is needed')
    

    parser.add_argument('--tensorboard_logdir', type=str, default=TENSORBOARD_LOGDIR, help='Directory for the tensorboard writer')
    parser.add_argument('--checkpoint_step', type=int, default=CHECKPOINT_STEP, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=VALIDATION_STEP, help='How often to perform validation (epochs)')
    parser.add_argument('--save_images', type=bool, default=SAVE_IMAGES, help='Indicate if it is necessary saving examples during validation')
    parser.add_argument('--save_images_step', type=bool, default=SAVE_IMAGES_STEP, help='How often save an image during validation')

    parser.add_argument('--save_model_path', type=str, default=SAVE_MODEL_PATH, help='path to save model')
    
    
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')

    args = parser.parse_args(params)

    return args


#------------------------------------------------------------------------------------------------------
#-------------------------------------------MAIN-------------------------------------------------------
#------------------------------------------------------------------------------------------------------


def main(params):
    """ Initialization and train launch """
    print(os.listdir())

    #-------------------------------Parse th arguments-------------------------------------------------
    args = get_arguments(params)
    
    #-------------------------------------end arguments-----------------------------------------------
   
    #------------------------------------Initialization-----------------------------------------------

    #Prepare the source and target sizes
    h, w = map(int, args.input_size_source.split(','))
    input_size_source = (h, w)

    h, w = map(int, args.input_size_target.split(','))
    input_size_target = (h, w)

    #Build the model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    #Build the Discirminator
    discriminator = LightDiscriminator(num_classes=args.num_classes) if args.light else  FCDiscriminator(num_classes=args.num_classes) 
    if torch.cuda.is_available() and args.use_gpu:                         
        discriminator = torch.nn.DataParallel(discriminator).cuda()                                                                           
    

    #Flops and paramters counter
    if args.flops:
        flops, parameters = parameter_flops_count(model, discriminator)
        
        print("*" * 20)
        print(f"Total number of operations: {round((flops.total()) / 1e+9, 4)}G FLOPS")
        print(f"Total number of parameters: {parameters}")
        print("*" * 20)


    #Load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(f"{args.pretrained_model_path}/latest_model.pth"))
        discriminator.module.load_state_dict(torch.load(f"{args.pretrained_model_path}/latest_discriminator.pth"))
        print('Done!')
    

    #Datasets instances 
    composed_source = transforms.Compose([transforms.ToTensor(),                                                               
                                    transforms.RandomHorizontalFlip(p=0.5),                                             
                                    transforms.RandomCrop(input_size_source, pad_if_needed=True)])

    composed_target = transforms.Compose([transforms.ToTensor(),                                                               
                                transforms.RandomHorizontalFlip(p=0.5),                                             
                                transforms.RandomCrop(input_size_target, pad_if_needed=True)])

    GTA5_dataset = GTA(root= args.data_source, 
                         images_folder= 'images', 
                         labels_folder= 'labels',
                         list_path= args.data_list_path_source,
                         info_file= args.info_file,
    )

    mask, weights = create_mask(GTA5_dataset.get_labels())
    

    Cityscapes_dataset_train = Cityscapes(root= args.data_target,
                         images_folder= 'images',
                         labels_folder='labels',
                         train=True,
                         info_file= args.info_file,
    )

    Cityscapes_dataset_val = Cityscapes(root= args.data_target,
                         images_folder= 'images',
                         labels_folder='labels',
                         train=False,
                         info_file= args.info_file,
    )

    #Dataloader instances
    trainloader = DataLoader(GTA5_dataset,
                            batch_size=args.batch_size,
                            shuffle=True, 
                            num_workers=args.num_workers,
                            pin_memory=True)  
    
    targetloader = DataLoader(Cityscapes_dataset_train,
                            batch_size=args.batch_size,
                            shuffle=True, 
                            num_workers=args.num_workers,
                            pin_memory=True)
    
    valloader = DataLoader(Cityscapes_dataset_val,
                            batch_size=1,
                            shuffle=False, 
                            num_workers=args.num_workers,
                            pin_memory=True)

        

    #Build Model Optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None
    
    #Build Discriminator Optimizer
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
  
    #------------------------------------end initialization-----------------------------------------------


    #--------------------------------------Train Launch---------------------------------------------------
    train(args, model, discriminator, optimizer, dis_optimizer, trainloader, targetloader, valloader, mask, weights)

    val(args, model, valloader, 'final')


#------------------------------------------------------------------------------------------------------
#-----------------------------------------END MAIN-----------------------------------------------------
#------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------
#------------------------------------------TRAIN-------------------------------------------------------
#------------------------------------------------------------------------------------------------------
def train(args, model, discriminator, optimizer, dis_optimizer, trainloader, targetloader, valloader, mask, weights):
    validation_run = 0 
    
    scaler = amp.GradScaler() 
    scaler_dis = amp.GradScaler()

    time = datetime.datetime.now(tz=timezone("Europe/Rome")).strftime("%d%B_%H:%M")
    suffix = f"{time}_{args.context_path}_light={args.light}_batch={args.batch_size}_lr={args.learning_rate}_croptarget({args.input_size_target})_cropsource({args.input_size_source})"
    args.save_model_path = args.save_model_path + suffix
    writer = SummaryWriter(f"{args.tensorboard_logdir}{suffix}")
    
    #Set the loss of G
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)

    #Set the loss of D
    bce_loss = torch.nn.BCEWithLogitsLoss()


    #Define the labels for adversarial training
    source_label = 0
    target_label = 1
    
    max_miou = 0
    step = 0

    for epoch in range(args.epoch_start_i, args.num_epochs):

        #Set the model to train mode
        model.train()

        #Adjust the G lr
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter = epoch, max_iter=args.num_epochs, power=args.power) 

        #Adjust the D lr
        lr_D = poly_lr_scheduler(dis_optimizer, args.learning_rate_D, iter = epoch, max_iter=args.num_epochs, power=args.power)

        #-----------------------------------------------------------------------------
        #Questo serve per tener traccia dello stato del training durante l'allenamento
        tq = tqdm(total = len(trainloader)*args.batch_size) #progress bar
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        #-----------------------------------------------------------------------------

        loss_seg_record = [] # array to store the value of the loss across the training
        loss_adv_record = []
        loss_D_record = []

        for i, ((source_images, source_labels), (target_images, _)) in enumerate(zip(trainloader, targetloader)):
            
            #----------------------------------Train G----------------------------------------------

            #Don't accumulate grads in D
            for param in discriminator.parameters():
                param.requires_grad = False

            
            #Train with source
            source_labels = source_labels.long()
            if torch.cuda.is_available() and args.use_gpu:
                source_images = source_images.cuda()
                source_labels = source_labels.cuda()
            
            optimizer.zero_grad()
            
            dis_optimizer.zero_grad() 

            with amp.autocast():
                output, output_sup1, output_sup2 = model(source_images) #final_output, output_x16down, output_(x32down*tail)

                if args.with_mask:
                    loss1 = NLLLoss(weight=weights.cuda(), ignore_index=255)(F.log_softmax(output) + torch.log(mask.cuda()), source_labels) # principal loss with mask
                else:
                    loss1 = loss_func(output, source_labels)                                               # principal loss without mask

                loss2 = loss_func(output_sup1, source_labels)       # loss with respect to output_x16down
                loss3 = loss_func(output_sup2, source_labels)       # loss with respect to output_(x32down*tail)
                loss_seg = loss1+loss2+loss3                        # The total loss is the sum of three terms (Equazione 2 sezione 3.3 del paper)

            scaler.scale(loss_seg).backward() 


            #Train with Target
            if torch.cuda.is_available() and args.use_gpu:
                target_images = target_images.cuda()

            with amp.autocast():
                output_target, _, _ = model(target_images) #Al discriminatore va passato solo output # TODO passare la maschera al target?

                D_out = discriminator(F.softmax(output_target))      

                loss_adv_target = bce_loss(D_out,
                                       torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda()) #Calcola la loss del D_out 
                                                                                                        #rispetto ad un tensore di source_label (quindi di 0) delle stesse dimensioni di D_out 
                                                                                                        #NB source_label != source_labels, source_label = 0 etichetta per con cui D distingue source e target
                                                                                                        #                                  source_labels = labels del batch di immagini provenienti da GTA      

                loss_adv = args.lambda_adv_target1 * loss_adv_target  #TODO controllare domani
            
            scaler.scale(loss_adv).backward()

            #----------------------------------end G-----------------------------------------------


            #----------------------------------Train D---------------------------------------------- 

            # bring back requires_grad
            for param in discriminator.parameters():
                param.requires_grad = True

            # train with source
            output = output.detach()

            with amp.autocast():
                D_out = discriminator(F.softmax(output) * mask.cuda()) ## @Edoardo, @Sebastiano - giusto?

                loss_D_source = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda())

            


            # train with target
            output_target = output_target.detach()

            with amp.autocast():
                D_out = discriminator(F.softmax(output_target))

                loss_D_target = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(target_label).cuda())

            #scaler_dis.scale().backward()
            loss_D = loss_D_source*0.5 + loss_D_target*0.5
            scaler_dis.scale(loss_D).backward() # Nuova backward

            #-----------------------------------end D-----------------------------------------------

            #Lo step degli optmizer va alla fine dei due training
            scaler.step(optimizer)
            scaler_dis.step(dis_optimizer)
            scaler.update()
            scaler_dis.update()

            tq.update(args.batch_size)
            tq.set_postfix({"loss_seg" : f'{loss_seg:.6f}', "loss_adv" : f'{loss_adv:.6f}', "loss_D" : f'{loss_D:.6f}'})
            step += 1
            writer.add_scalar('loss_seg_step', loss_seg, step)
            writer.add_scalar('loss_adv_step', loss_adv, step)
            writer.add_scalar('loss_D_step', loss_D, step)
            loss_seg_record.append(loss_seg.item())
            loss_adv_record.append(loss_adv.item())
            loss_D_record.append(loss_D.item())

    
        tq.close()
        #Loss_seg
        loss_train_seg_mean = np.mean(loss_seg_record)
        writer.add_scalar('epoch/loss_epoch_train_seg', float(loss_train_seg_mean), epoch)
        print(f'Average loss_seg for epoch {epoch}: {loss_train_seg_mean}')
        #Loss_adv
        loss_train_adv_mean = np.mean(loss_adv_record)
        writer.add_scalar('epoch/loss_epoch_train_adv', float(loss_train_adv_mean), epoch)
        print(f'Average loss_adv for epoch {epoch}: {loss_train_adv_mean}')
        #Loss_D
        loss_train_D_mean = np.mean(loss_D_record)
        writer.add_scalar('epoch/loss_epoch_train_D', float(loss_train_D_mean), epoch)
        print(f'Average loss_D for epoch {epoch}: {loss_train_D_mean}')

        if epoch % args.checkpoint_step == 0 and epoch != 0:
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest_model.pth'))
            torch.save(discriminator.module.state_dict(), os.path.join(args.save_model_path, 'latest_discriminator.pth'))
        
        if epoch % args.validation_step == 0 and epoch != 0:
                precision, miou = val(args, model, valloader, validation_run)
                validation_run += 1
                if miou > max_miou:
                    max_miou = miou
                    os.makedirs(args.save_model_path, exist_ok=True)
                    torch.save(model.module.state_dict(),
                            os.path.join(args.save_model_path, 'best_model.pth'))
                writer.add_scalar('epoch/precision_val', precision, epoch)
                writer.add_scalar('epoch/miou val', miou, epoch)


def val(args, model, dataloader, validation_run):

    print(f"{'#'*10} VALIDATION {'#' * 10}")

    # label_info = get_label_info(csv_path)

    #prepare info_file to save examples
    info = json.load(open(args.data_target+"/"+args.info_file))
    palette = {i if i!=19 else 255:info["palette"][i] for i in range(20)}
    mean = torch.as_tensor(info["mean"])
    if torch.cuda.is_available() and args.use_gpu:
        mean = mean.cuda() 

    with torch.no_grad():
        model.eval() #set the model in the evaluation mode
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes)) #create a square arrey with side num_classes
        for i, (image, label) in enumerate(tqdm(dataloader)): #get a batch of data and the respective label at each iteration
            label = label.type(torch.LongTensor) #set the type of the label to long
            label = label.long()
            if torch.cuda.is_available() and args.use_gpu:
                image = image.cuda()
                label = label.cuda()

            #get RGB predict image
            predict = model(image).squeeze() #remove all the dimension equal to one => For example, if input is of shape: (A×1×B×C×1×D) then the out tensor will be of shape: (A×B×C×D)
            
            #--------------------------------------------------------------------------
            # Verificare che i layer di predict sono nello stesso ordine della maschera
            # TODO
            #---------------------------------------------------------------------------
            
            predict = reverse_one_hot(predict) #from one_hot_encoding to class key?
            predict = np.array(predict.cpu()) #move predict to cpu and convert it into a numpy array

            #get RGB label image
            label = label.squeeze()
            if args.loss == 'dice':#check what loss is being used
                label = reverse_one_hot(label)
            label = np.array(label.cpu())

            #compute per pixel accuracy
            precision = compute_global_accuracy(predict, label) #accuracy of the prediction
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes) 
            
            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            path_to_save= args.save_model_path+f"/val_results/{validation_run}"

            if args.save_images and i % args.save_images_step == 0 : 
                index_image = get_index(int(i/args.save_images_step))
                os.makedirs(path_to_save, exist_ok=True)
                save_images(mean, palette, image, predict, label, 
                path_to_save+"/"+index_image+".png") 
            
            precision_record.append(precision)
           
    
    precision = np.mean(precision_record)
    miou_list = per_class_iu(hist) #come funziona questo metodo?
    miou = np.mean(miou_list)

    print('precision per pixel for test: %.3f' % precision)
    print('mIoU for validation: %.3f' % miou)
    print(f'mIoU per class: {miou_list}')

    return precision, miou
    



if __name__ == '__main__':
    params = [
        '--epoch_start_i', '0',
        '--checkpoint_step', '7',
        '--validation_step', '7',
        '--num_epochs', '50',
        '--learning_rate', '2.5e-2',
        '--data_target', '/content/drive/MyDrive/MLDL_Project/AdaptSegNet/data/Cityscapes', 
        '--data_source', '/content/drive/MyDrive/MLDL_Project/AdaptSegNet/data/GTA5', 
        '--num_workers', '8',
        '--num_classes', '19',
        '--cuda', '0',
        '--batch_size', '6',
        '--save_model_path', '/content/drive/MyDrive/MLDL_Project/PriorNet/models/',
        '--tensorboard_logdir', '/content/drive/MyDrive/MLDL_Project/PriorNet/runs/',
        '--context_path', 'resnet101',  # set resnet18 or resnet101, only support resnet18 and resnet101
        '--optimizer', 'sgd',

    ]

    main(params)
    
    # args = get_arguments(params)

    # model = BiSeNet(19, 'resnet101')


    # h, w = map(int, args.input_size_target.split(','))
    # input_size_target = (h, w)

    # composed_target = transforms.Compose([transforms.ToTensor(),                                                               
    #                             transforms.RandomHorizontalFlip(p=0.5),                                             
    #                             transforms.RandomCrop(input_size_target, pad_if_needed=True)])

    # Cityscapes_dataset_val = Cityscapes(root= args.data_target,
    #                      images_folder= 'images',
    #                      labels_folder='labels',
    #                      train=False,
    #                      info_file= args.info_file,
    #                      transforms= composed_target
    # )

    # valloader = DataLoader(Cityscapes_dataset_val,
    #                         batch_size=1,
    #                         shuffle=False, 
    #                         num_workers=args.num_workers,
    #                         pin_memory=True)


    # val(args, model, valloader, 0)


    

