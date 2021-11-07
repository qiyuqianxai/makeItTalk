import codeop
import json
import re
import sys
sys.path.append('thirdparty/AdaptiveWingLoss')
import os, glob
import numpy as np
import cv2
import argparse
from src.approaches.train_image_translation import Image_translation_block
import torch
import pickle
import face_alignment
from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
import shutil
import util.utils as util
from scipy.signal import savgol_filter
from time import sleep
from src.approaches.train_audio2landmark import Audio2landmark_model

default_head_name = 'dwayne2'
default_audio = 'intro'
ADD_NAIVE_EYE = True
CLOSE_INPUT_FACE_MOUTH = False

parser = argparse.ArgumentParser()
parser.add_argument('--jpg', type=str, default='{}.jpg'.format(default_head_name))
parser.add_argument('--audio', type=str, default='{}.mp3'.format(default_audio))
parser.add_argument('--close_input_face_mouth', default=CLOSE_INPUT_FACE_MOUTH, action='store_true')

parser.add_argument('--load_AUTOVC_name', type=str, default='examples/ckpt/ckpt_autovc.pth')
parser.add_argument('--load_a2l_G_name', type=str, default='examples/ckpt/ckpt_speaker_branch.pth')
parser.add_argument('--load_a2l_C_name', type=str, default='examples/ckpt/ckpt_content_branch.pth') #ckpt_audio2landmark_c.pth')
parser.add_argument('--load_G_name', type=str, default='examples/ckpt/ckpt_116_i2i_comb.pth') #ckpt_image2image.pth') #ckpt_i2i_finetune_150.pth') #c

parser.add_argument('--amp_lip_x', type=float, default=2.)
parser.add_argument('--amp_lip_y', type=float, default=2.)
parser.add_argument('--amp_pos', type=float, default=.5)
parser.add_argument('--reuse_train_emb_list', type=str, nargs='+', default=[]) #  ['iWeklsXc0H8']) #['45hn7-LXDX8']) #['E_kmpT-EfOg']) #'iWeklsXc0H8', '29k8RtSUjE0', '45hn7-LXDX8',
parser.add_argument('--add_audio_in', default=False, action='store_true')
parser.add_argument('--comb_fan_awing', default=False, action='store_true')
parser.add_argument('--output_folder', type=str, default='examples')
parser.add_argument('--output', type=str, default='output.mp4')

parser.add_argument('--test_end2end', default=True, action='store_true')
parser.add_argument('--dump_dir', type=str, default='', help='')
parser.add_argument('--pos_dim', default=7, type=int)
parser.add_argument('--use_prior_net', default=True, action='store_true')
parser.add_argument('--transformer_d_model', default=32, type=int)
parser.add_argument('--transformer_N', default=2, type=int)
parser.add_argument('--transformer_heads', default=2, type=int)
parser.add_argument('--spk_emb_enc_size', default=16, type=int)
parser.add_argument('--init_content_encoder', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--reg_lr', type=float, default=1e-6, help='weight decay')
parser.add_argument('--write', default=False, action='store_true')
parser.add_argument('--segment_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--emb_coef', default=3.0, type=float)
parser.add_argument('--lambda_laplacian_smooth_loss', default=1.0, type=float)
parser.add_argument('--use_11spk_only', default=False, action='store_true')

args = parser.parse_args()

def generate():
    ''' STEP 1: preprocess input single image '''
    img = cv2.imread(args.jpg)
    img = cv2.resize(img, (256, 256))
    predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True)
    shapes = predictor.get_landmarks(img)
    if (not shapes or len(shapes) != 1):
        print('Cannot detect face landmarks. Exit.')
        exit(-1)
    shape_3d = shapes[0]

    if (args.close_input_face_mouth):
        util.close_input_face_mouth(shape_3d)

    ''' Additional manual adjustment to input face landmarks (slimmer lips and wider eyes) '''
    # shape_3d[48:, 0] = (shape_3d[48:, 0] - np.mean(shape_3d[48:, 0])) * 0.95 + np.mean(shape_3d[48:, 0])
    shape_3d[49:54, 1] += 1.
    shape_3d[55:60, 1] -= 1.
    shape_3d[[37, 38, 43, 44], 1] -= 2
    shape_3d[[40, 41, 46, 47], 1] += 2

    ''' STEP 2: normalize face as input to audio branch '''
    shape_3d, scale, shift = util.norm_input_face(shape_3d)

    ''' STEP 3: Generate audio data as input to audio branch '''
    # audio real data
    au_data = []
    au_emb = []
    ains = [args.audio]
    for ain in ains:
        os.system('ffmpeg -y -loglevel error -i {} -ar 16000 examples/tmp.wav'.format(ain))
        shutil.copyfile('examples/tmp.wav', '{}'.format(ain))

        # au embedding
        from thirdparty.resemblyer_util.speaker_emb import get_spk_emb
        me, ae = get_spk_emb('{}'.format(ain))
        au_emb.append(me.reshape(-1))

        # print('Processing audio file', ain)
        c = AutoVC_mel_Convertor('examples')

        au_data_i = c.convert_single_wav_to_autovc_input(audio_filename=ain,
                                                         autovc_model_path=args.load_AUTOVC_name)
        au_data += au_data_i

    if os.path.isfile('examples/tmp.wav'):
        os.remove('examples/tmp.wav')

    # landmark fake placeholder
    fl_data = []
    rot_tran, rot_quat, anchor_t_shape = [], [], []
    for au, info in au_data:
        au_length = au.shape[0]
        fl = np.zeros(shape=(au_length, 68 * 3))
        fl_data.append((fl, info))
        rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
        rot_quat.append(np.zeros(shape=(au_length, 4)))
        anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))

    if (os.path.exists(os.path.join('examples', 'dump', 'random_val_fl.pickle'))):
        os.remove(os.path.join('examples', 'dump', 'random_val_fl.pickle'))
    if (os.path.exists(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))):
        os.remove(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))
    if (os.path.exists(os.path.join('examples', 'dump', 'random_val_au.pickle'))):
        os.remove(os.path.join('examples', 'dump', 'random_val_au.pickle'))
    if (os.path.exists(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))):
        os.remove(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))

    with open(os.path.join('examples', 'dump', 'random_val_fl.pickle'), 'wb') as fp:
        pickle.dump(fl_data, fp)
    with open(os.path.join('examples', 'dump', 'random_val_au.pickle'), 'wb') as fp:
        pickle.dump(au_data, fp)
    with open(os.path.join('examples', 'dump', 'random_val_gaze.pickle'), 'wb') as fp:
        gaze = {'rot_trans': rot_tran, 'rot_quat': rot_quat, 'anchor_t_shape': anchor_t_shape}
        pickle.dump(gaze, fp)

    ''' STEP 4: RUN audio->landmark network'''
    # print('step4')
    # if model is None:
    model = Audio2landmark_model(args, jpg_shape=shape_3d)
    print('model load')
    if (len(args.reuse_train_emb_list) == 0):
        model.test(au_emb=au_emb, args=args)
    else:
        model.test(au_emb=None)
    # print('model test audio embedding')
    ''' STEP 5: de-normalize the output to the original image scale '''
    # print('step5')
    fls = glob.glob1('examples', 'pred_fls_*.txt')
    fls.sort()

    for i in range(0, len(fls)):
        fl = np.loadtxt(os.path.join('examples', fls[i])).reshape((-1, 68, 3))
        fl[:, :, 0:2] = -fl[:, :, 0:2]
        fl[:, :, 0:2] = fl[:, :, 0:2] / scale - shift

        if (ADD_NAIVE_EYE):
            fl = util.add_naive_eye(fl)

        # additional smooth
        fl = fl.reshape((-1, 204))
        fl[:, :48 * 3] = savgol_filter(fl[:, :48 * 3], 15, 3, axis=0)
        fl[:, 48 * 3:] = savgol_filter(fl[:, 48 * 3:], 5, 3, axis=0)
        fl = fl.reshape((-1, 68, 3))

        ''' STEP 6: Imag2image translation '''
        # print('step6')
        model = Image_translation_block(args, single_test=True)
        with torch.no_grad():
            model.single_test(jpg=img, fls=fl, filename=fls[i], prefix=args.jpg.split('.')[0], args=args)
            print('finish image2image gen')
        os.remove(os.path.join('examples', fls[i]))


user_image_dir = '/workspace/go_proj/src/Ai_WebServer/static/algorithm/MakeItTalk/user_imgs'
user_audio_dir = '/workspace/go_proj/src/Ai_WebServer/static/algorithm/MakeItTalk/user_audios'
result_dir = '/workspace/go_proj/src/Ai_WebServer/static/algorithm/MakeItTalk/res_videos'
message_json = '/workspace/go_proj/src/Ai_WebServer/algorithm_utils/MakeItTalk/message.json'


def set_args(message):
    args.jpg = os.path.join(user_image_dir, message['user_img'])
    args.audio = os.path.join(user_audio_dir, message['user_audio'])
    args.output = os.path.join(result_dir, re.sub('\.jpg|\.jpeg|\.png|\.mp3',"",message['user_img']+"_"+message['user_audio'])+"_"+
                               str(message['amp_lip_x'])+str(message['amp_lip_y'])+str(message['amp_pos'])+".mp4")
    args.amp_lip_x = float(message['amp_lip_x'])  # 嘴唇横向摆动幅度：0-5（浮点数）
    args.amp_lip_y = float(message['amp_lip_y'])  # 嘴唇纵向摆动幅度：0-5（浮点数）
    args.amp_pos = float(message['amp_pos'])  # 头部摆动幅度：0-1（浮点数）

import os
if __name__ == '__main__':
    last_message = {}
    while True:

            with open(message_json, "r", encoding="utf-8") as f:
                message = json.load(f)

            if message == last_message:
                print('waiting')
                sleep(1)
                continue
            else:
                set_args(message)
                last_message = message
            if os.path.exists(args.output):
                print("MakeItTalk exist...")
                sleep(1)
                continue
            generate()
        # except Exception as e:
        #     print(e)
        #     sleep(1)
        #     continue
