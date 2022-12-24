# Evaluate
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 讀取資料庫
import os
import shutil
import pandas as pd
from glob import glob
from torch.autograd import Variable

#from tabulate import tabulate
import random
import numpy as np
from scipy import stats
import heapq
import time
from pdb import set_trace as bp

# Homework -- 以 "inference-npy" inference 資料庫並計算準確率
import os
import shutil
import pandas as pd
from glob import glob
from torch.autograd import Variable


# Convert flac to wav
from glob import glob
import os
from pydub import AudioSegment
from tqdm import tqdm

# Feature Extraction
import math
import logging
# extract fbanck from wav and save to file
# pre processd an audio in 0.09912s

import os
import librosa
import numpy as np
import pandas as pd
from glob import glob
from python_speech_features import fbank, delta
from tqdm import tqdm
from time import time

# Model & Loss Function
import torch
import torch.nn as nn
import math
from torch.autograd import Function

# Homework -- stochastic_mini_batch
import numpy as np
import pandas as pd
from os import walk

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

# Train
import os
import sys
import numpy as np
import random

import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchsummary import summary

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Setting Constants
# 設定超參數
WAV_DIR = './LibriSpeech/train-clean-100/'
DATASET_DIR = './LibriSpeech/train-clean-100-npy/'

BATCH_SIZE = 32
TRIPLET_PER_BATCH = 3

NUM_FRAMES = 160
SAMPLE_RATE = 16000
TRUNCATE_SOUND_SECONDS = (0.2, 1.81)  # (start_sec, end_sec)
NUM_SPEAKERS = 251
EMBEDDING_SIZE = 512

# Feature Extraction
class SilenceDetector(object):
    def __init__(self, threshold=20, bits_per_sample=16):
        self.cur_SPL = 0
        self.threshold = threshold
        self.bits_per_sample = bits_per_sample
        self.normal = pow(2.0, bits_per_sample - 1)
        self.logger = logging.getLogger('balloon_thrift')

    def is_silence(self, chunk):
        self.cur_SPL = self.soundPressureLevel(chunk)
        is_sil = self.cur_SPL < self.threshold
        # print('cur spl=%f' % self.cur_SPL)
        if is_sil:
            self.logger.debug('cur spl=%f' % self.cur_SPL)
        return is_sil

    def soundPressureLevel(self, chunk):
        value = math.pow(self.localEnergy(chunk), 0.5)
        value = value / len(chunk) + 1e-12
        value = 20.0 * math.log(value, 10)
        return value

    def localEnergy(self, chunk):
        power = 0.0
        for i in range(len(chunk)):
            sample = chunk[i] * self.normal
            power += sample*sample
        return power


# Feature Extraction
def find_files(directory, pattern='**/*.wav'):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)


def VAD(audio):
    chunk_size = int(SAMPLE_RATE * 0.05)  # 50ms
    index = 0
    sil_detector = SilenceDetector(15)
    nonsil_audio = []
    while index + chunk_size < len(audio):
        if not sil_detector.is_silence(audio[index: index + chunk_size]):
            nonsil_audio.extend(audio[index: index + chunk_size])
        index += chunk_size

    return np.array(nonsil_audio)


def read_audio(filename, sample_rate=SAMPLE_RATE):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    # print(audio.shape[0])
    audio = VAD(audio.flatten())
    # print(audio.shape[0])
    start_sec, end_sec = TRUNCATE_SOUND_SECONDS
    start_frame = int(start_sec * SAMPLE_RATE)
    end_frame = int(end_sec * SAMPLE_RATE)

    if len(audio) < (end_frame - start_frame):
        au = [0] * (end_frame - start_frame)
        for i in range(len(audio)):
            au[i] = audio[i]
        audio = np.array(au)
    # print(audio.shape[0])
    return audio


def normalize_frames(m, epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v), epsilon) for v in m]


def extract_features(signal=np.random.uniform(size=48000), target_sample_rate=SAMPLE_RATE):
    filter_banks, energies = fbank(signal, samplerate=target_sample_rate, nfilt=64,
                                   winlen=0.025)  # filter_bank (num_frames , 64),energies (num_frames ,)
    filter_banks = normalize_frames(filter_banks)
    frames_features = filter_banks  # (num_frames , 64)
    num_frames = len(frames_features)  # (num_frames)
    return np.reshape(np.array(frames_features), (num_frames, 64, 1))  # (num_frames,64, 1)


def data_catalog(dataset_dir=DATASET_DIR, pattern='*.npy'):
    libri = pd.DataFrame()
    libri['filename'] = find_files(dataset_dir, pattern=pattern)
    libri['filename'] = libri['filename'].apply(lambda x: x.replace('\\', '/'))  # normalize windows paths
    libri['speaker_id'] = libri['filename'].apply(lambda x: x.split('/')[-1].split('-')[0])  # x.split('/')[-1]->1-100-0001.wav
    num_speakers = len(libri['speaker_id'].unique())
    print('Found {} files with {} different speakers.'.format(str(len(libri)).zfill(7), str(num_speakers).zfill(5)))
    return libri
    #                          filename                                       speaker_id
    #   0    LibriSpeech/train-clean-100/1/100/1-100-0001.wav        1
    #   1    LibriSpeech/train-clean-100/1/100/1-100-0002.wav        1


def prep(libri, out_dir=DATASET_DIR):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # i=0
    for i in tqdm(range(len(libri))):
        filename = libri[i:i + 1]['filename'].values[0]  # for example: LibriSpeech/train-clean-100/1/100/1-100-0001.wav
        target_filename = out_dir + filename.split("/")[-1].split('.')[
            0] + '.npy'  # for example: LibriSpeech/train-clean-100-npy/1-100-0001.npy
        # 確認是否已經產生.npy檔了
        if os.path.exists(target_filename):
            continue

        fp = open(target_filename, 'w')
        fp.close()
        raw_audio = read_audio(filename)
        feature = extract_features(raw_audio, target_sample_rate=SAMPLE_RATE)
        if feature.ndim != 3 or feature.shape[0] < NUM_FRAMES or feature.shape[1] != 64 or feature.shape[2] != 1:
            print('there is an error in file:', filename)
            continue
        np.save(target_filename, feature)


def preprocess_and_save(wav_dir=WAV_DIR, out_dir=DATASET_DIR):
    orig_time = time()
    libri = data_catalog(wav_dir, pattern='**/*.wav')

    print("Extract fbank from audio and save as npy")
    prep(libri, out_dir)
    print("Extract audio features and save it as npy file, cost {0} seconds".format(time() - orig_time))


# Model & Loss Function
class PairwiseDistance(Function):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        # assert x1.size() == x2.size()
        # x1 = torch.unsqueeze(x1, 1) # 30 1 512
        # x2 = torch.unsqueeze(x2, 1)
        # diff = torch.bmm(x1.view(x1.shape[0],1,512),x2.view(x2.shape[0],512,1))
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)
        # return diff


class TripletMarginLoss(Function):
    """Triplet loss function.
    """

    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)  # norm 2

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, negative)
        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.sum(dist_hinge)
        return loss


class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
               + inplace_str + ')'


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class myResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):

        super(myResNet, self).__init__()

        self.relu = ReLU(inplace=True)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.inplanes = 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.inplanes = 256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.inplanes = 512
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.layer4 = self._make_layer(block, 512, layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


class DeepSpeakerModel(nn.Module):
    def __init__(self, embedding_size, num_classes, feature_dim=64):
        super(DeepSpeakerModel, self).__init__()

        self.embedding_size = embedding_size
        self.model = myResNet(BasicBlock, [1, 1, 1, 1])
        if feature_dim == 64:
            self.model.fc = nn.Linear(512 * 4, self.embedding_size)
        elif feature_dim == 40:
            self.model.fc = nn.Linear(256 * 5, self.embedding_size)
        self.model.classifier = nn.Linear(self.embedding_size, num_classes)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)

        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer2(x)

        x = self.model.conv3(x)
        x = self.model.bn3(x)
        x = self.model.relu(x)
        x = self.model.layer3(x)

        x = self.model.conv4(x)
        x = self.model.bn4(x)
        x = self.model.relu(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.model.fc(x)
        self.features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha = 1
        self.features = self.features * alpha

        # x = x.resize(int(x.size(0) / 17),17 , 512)
        # self.features =torch.mean(x,dim=1)
        # x = self.model.classifier(self.features)
        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res


# Homework -- stochastic_mini_batch
class stochastic_mini_batch(Dataset):

    def __init__(self, libri, train_dict, unique_speakers_list):
        self.libri = libri
        self.train_dict = train_dict
        self.unique_speakers_list = unique_speakers_list

    def clipped_audio(self, x, num_frames=NUM_FRAMES):
        if x.shape[0] > num_frames:
            bias = np.random.randint(0, x.shape[0] - num_frames)
            clipped_x = x[bias: num_frames + bias]
        else:
            clipped_x = x

        return clipped_x

    def __len__(self):
        return len(self.libri)

    def __getitem__(self, index):

        ################ you should write here ################
        # hint
        # 1. sample anchor file ,positive file, negative file
        # 2. np.load(...)
        # 3. clipped_audio(...)
        # 4. torch.from_numpy(....transpose ((2, 0, 1)))

        anchor_file_name = self.libri.iloc[index, 0]
        anchor_label = self.libri.iloc[index, 1]

        # search positive_file, 若找到和anchor file相同,則以下一個index當positive file
        # if true ==> 該id至少有兩個file
        positive_file_name = anchor_file_name
        positive_label = anchor_label
        if str(anchor_label) in self.train_dict:
            file_nums = len(self.train_dict[anchor_label])
            pos_idx = random.randint(0, file_nums-1)
            if anchor_file_name == self.train_dict[anchor_label][pos_idx]:
                pos_idx = (pos_idx + 1) % file_nums  # circular list
            positive_file_name = self.train_dict[anchor_label][pos_idx]

        # search negative_file and negative_label, 若找到和anchor id相同,則以下一個index當negative id
        id_nums = len(self.unique_speakers_list)
        idx = random.randint(0, id_nums-1)
        if self.unique_speakers_list[idx] == anchor_label:
            idx = (idx + 1) % id_nums  # circular list
        negative_label = self.unique_speakers_list[idx]
        file_nums = len(self.train_dict[str(negative_label)])
        neg_idx = random.randint(0, file_nums-1)
        negative_file_name = self.train_dict[str(negative_label)][neg_idx]

        # load npy files
        # data type : float64
        anchor_file = np.load(anchor_file_name)
        positive_file = np.load(positive_file_name)
        negative_file = np.load(negative_file_name)

        # clip audio
        anchor_file = self.clipped_audio(anchor_file)
        positive_file = self.clipped_audio(positive_file)
        negative_file = self.clipped_audio(negative_file)
        #print(f"type(anchor_file):{type(anchor_file)}\ttype(positive_file):{type(positive_file)}\ttype(negative_file):{type(negative_file)}")
        #print(f"anchor_file.shape:{anchor_file.shape}\tpositive_file.shape:{positive_file.shape}\tnegative_file.shape:{negative_file.shape}")
        #print(f"anchor_file.dtype):{anchor_file.dtype}\tpositive_file.dtype:{positive_file.dtype}\tnegative_file.dtype:{negative_file.dtype}")

        # convert to torch and do transpose
        anchor_file = torch.from_numpy(anchor_file)
        anchor_file = torch.transpose(anchor_file, 0, 1)
        anchor_file = torch.transpose(anchor_file, 0, 2)
        positive_file = torch.from_numpy(positive_file)
        positive_file = torch.transpose(positive_file, 0, 1)
        positive_file = torch.transpose(positive_file, 0, 2)
        negative_file = torch.from_numpy(negative_file)
        negative_file = torch.transpose(negative_file, 0, 1)
        negative_file = torch.transpose(negative_file, 0, 2)

        #print(f"type(anchor_file):{type(anchor_file)}\ttype(positive_file):{type(positive_file)}\ttype(negative_file):{type(negative_file)}")
        #print(f"anchor_file.shape:{anchor_file.shape}\tpositive_file.shape:{positive_file.shape}\tnegative_file.shape:{negative_file.shape}")

        return anchor_file, positive_file, negative_file, anchor_label, positive_label, negative_label


# Train
def create_dict(files, labels, spk_uniq):
    train_dict = {}
    for i in range(len(spk_uniq)):
        train_dict[spk_uniq[i]] = []
    for i in range(len(labels)):
        train_dict[labels[i]].append(files[i])
    for spk in spk_uniq:
        if len(train_dict[spk]) < 2:
            train_dict.pop(spk)

    unique_speakers = list(train_dict.keys())
    return train_dict, unique_speakers


def load_model(model_path):
    model = DeepSpeakerModel(embedding_size=EMBEDDING_SIZE, num_classes=NUM_SPEAKERS)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)

    print('=> loading checkpoint')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    return model, optimizer


def train(model, train_loader, optimizer):
    epoch = 0
    model.cuda()
    summary(model, input_size=(1, 160, 64))
    for epoch in range(100):
        model.train()
        for batch_idx, (data_a, data_p, data_n, label_a, label_p, label_n) in tqdm(enumerate(train_loader)):
            data_a, data_p, data_n = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor), data_n.type(torch.FloatTensor)
            data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()
            data_a, data_p, data_n = Variable(data_a), Variable(data_p), Variable(data_n)
            out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)

            triplet_loss = TripletMarginLoss(0.2).forward(out_a, out_p, out_n)
            loss = triplet_loss
            # compute gradient and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('selected_triplet_loss', triplet_loss.data)
        print("epoch:", epoch)
        # torch.save(model.state_dict(),"checkpoint_{}.pt".format(epoch))
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   "checkpoint_{}.pt".format(epoch))


def init():
    # Convert flac to wav
    if not os.path.exists(DATASET_DIR):
        files = glob(os.path.join(WAV_DIR, '**/**/*.flac'), recursive=True)

        for file in tqdm(files):
            tmp = AudioSegment.from_file(file, format='flac')
            tmp.export(file.replace('.flac', '.wav'), format='wav')

    # Feature Extraction
    np.set_printoptions(threshold=10)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('max_colwidth', 100)
    preprocess_and_save(WAV_DIR, DATASET_DIR)


def do_training():
    # Train
    print('Looking for fbank features [.npy] files in {}.'.format(DATASET_DIR))
    libri = data_catalog(DATASET_DIR)
    #  index   filename                                 speaker_id
    #   0    audio/LibriSpeechSamples/train-clean-100-npy/1-100-0001.npy        1
    #   1    audio/LibriSpeechSamples/train-clean-100-npy/1-100-0002.npy        1

    #print(libri.iloc[0,0])
    unique_speakers = libri['speaker_id'].unique()  # 251 speaker

    # train_dict ==> 得到每個id對應到的檔案
    # train_dict只有儲存至少兩個以上檔案的id
    train_dict, unique_speakers_list = create_dict(libri['filename'], libri['speaker_id'], unique_speakers)

    train_dir = stochastic_mini_batch(libri, train_dict, unique_speakers_list)

    train_loader = DataLoader(train_dir, batch_size=BATCH_SIZE, shuffle=True)



    # Retraining
    ckpt_path = "./pretrained/checkpoint_78.pt"
    if ckpt_path != "":
        model, optimizer = load_model(ckpt_path)
    else:
        model = DeepSpeakerModel(embedding_size=EMBEDDING_SIZE, num_classes=NUM_SPEAKERS)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)

    train(model, train_loader, optimizer)


# Evaluate
class Database():
    "Simulated data structure"

    def __init__(self, data_num):
        self.embs = np.ndarray((data_num, 512), dtype=float)
        self.labels = []
        self.indices = 0

    def __len__(self):
        return self.indices

    def insert(self, label, emb, index=None):
        " Insert testing data "

        self.embs[self.indices] = emb
        self.labels.append(label)
        self.indices += 1

    def get_most_similar(self, embTest):
        testTiles = np.tile(embTest, (self.indices, 1))
        similarities = np.sum(testTiles * self.embs[0:self.indices], axis=1)
        max_similarity = np.max(similarities)
        max_id = np.argmax(similarities)
        return max_id, max_similarity, self.embs[max_id]

    def get_label_by_id(self, id):
        return self.labels[id]


def get_similarity(embA, embB):  # inner product
    ans = np.sum(embA * embB)
    return ans


def do_evaluation():
    # Evaluate
    model = DeepSpeakerModel(embedding_size=512, num_classes=251)
    # Load your model
    checkpoint = torch.load("./pretrained/checkpoint_66.pt", map_location='cpu')
    model.load_state_dict(checkpoint, False)

    # 讀取資料庫
    libri = data_catalog("database-npy", pattern='*.npy')
    new_x = []
    labels = []
    model.eval()

    with torch.no_grad():
        flag = 0

        for i in range(int(len(libri))):
            new_x = []
            filename = libri[i:i + 1]['filename'].values[0]
            filelabel = libri[i:i + 1]['speaker_id'].values[0]
            x = np.load(filename)
            if (x.shape[0] > 160):
                for bias in range(0, x.shape[0] - 160, 160):
                    clipped_x = x[bias:bias + 160]
                    new_x.append(clipped_x)
                    labels.append(filelabel)
            else:
                clipped_x = x
                new_x.append(clipped_x)
                labels.append(filelabel)

            x = np.array(new_x)
#            print(x.shape)
            x_tensor = Variable(torch.from_numpy(x.transpose((0, 3, 1, 2))).type(torch.FloatTensor).contiguous())
#            print(x_tensor.shape)
            embedding = model(x_tensor)
            if i == 0:
                temp_embedding = embedding
            else:
                temp_embedding = torch.cat((temp_embedding, embedding), 0)

        temp_embedding = temp_embedding.cpu().detach().numpy()
        labels = np.array(labels)
        labels = labels.astype("int32")
#        print(labels.shape)
#        print(temp_embedding.shape)
        np.save('emb', temp_embedding)
        np.save('emb_label', labels)

    database = Database(20000)

    for i in range(len(labels)):
        test_array, test_label = temp_embedding[i], labels[i]
        database.insert(test_label, test_array)
    print("inserting database completed")


    # Homework -- 以 "inference-npy" inference 資料庫並計算準確率
    libri = data_catalog("inference-npy", pattern='*.npy')  # audio/LibriSpeechTest/test-clean-npy
    infer_labels = []
    ########## you should write here ########
    # clipped
    # model and concat
    with torch.no_grad():
        flag = 0

        for i in range(int(len(libri))):
            new_x = []
            filename = libri[i:i + 1]['filename'].values[0]
            filelabel = libri[i:i + 1]['speaker_id'].values[0]
            x = np.load(filename)
            if (x.shape[0] > 160):
                for bias in range(0, x.shape[0] - 160, 160):
                    clipped_x = x[bias:bias + 160]
                    new_x.append(clipped_x)
                    infer_labels.append(filelabel)
            else:
                clipped_x = x
                new_x.append(clipped_x)
                infer_labels.append(filelabel)

            x = np.array(new_x)
            #print(x.shape)
            x_tensor = Variable(torch.from_numpy(x.transpose((0, 3, 1, 2))).type(torch.FloatTensor).contiguous())
            #print(x_tensor.shape)
            embedding = model(x_tensor)
            if i == 0:
                infer_embedding = embedding
            else:
                infer_embedding = torch.cat((infer_embedding, embedding), 0)

    #print(f"len(infer_labels): {len(infer_labels)} len(new_x): {len(new_x)}")
    correct = 0
    for i in range(len(infer_labels)):
        embTest = infer_embedding[i]
        #print(f"embTest.shape: {embTest.shape}")
        max_id, max_similarity, embs = database.get_most_similar(embTest)
        predicted_label = database.get_label_by_id(max_id)
        #print(f"label: {infer_labels[i]} predict: {predicted_label}")
        #print(f"type(infer_labels[i]): {type(infer_labels[i])} type(predicted_label): {type(predicted_label)}")
        if str(infer_labels[i]) == str(predicted_label):
            correct = correct + 1
    print(f"correct : {correct/len(infer_labels)}")







# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 0: training  1: evaluation
    Status = 0

    if Status == 0:
        do_training()
        init()
    else:
        do_evaluation()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
