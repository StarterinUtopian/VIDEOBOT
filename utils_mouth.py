import os
import torch
import numpy as np
import sys
import wave
from .roman2index import roman2index
import cv2

def initialize_tacotron(checkpoint_path, hparams, load_model, Denoiser):

    waveglow_path = os.getcwd()+'/checkpoints/waveglow_256channels_ljs_v3.pt'

    hparams.n_mel_channels = 80
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])

    model.cuda().eval().half()
    waveglow = torch.load(waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow).cuda()
    return model, waveglow, denoiser


def save_wav(audio_denoised, save_audio_path, samplerate=44100):
    t = np.linspace(0., 1., samplerate)
    amplitude = np.iinfo(np.int16).max
    datas = audio_denoised/max(audio_denoised)

    datas=(datas*amplitude).astype(np.int16)

    #waveで出力する
    w = wave.Wave_write(save_audio_path)
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(22050)
    w.writeframes(datas)
    w.close()


def get_index(text, alignments):
    kana_index = roman2index(text)
    kana_index_array = np.array(kana_index)

    n_word = "b,p,m,by,py,my".split(",")
    i_word = "s,t,n,r,z,d,ky,sh,ch,ny,hy,ry,gy,j,y,ts".split(",")
    u_word = "w,f,v".split(",")
    v_word = "k,h,g".split(",")

    kana_list = []
    word_list = []
    kana_list.append(text[:min(kana_index_array)])
    for i in range(len(kana_index_array)-1):
        kana_list.append(text[kana_index_array[i]:kana_index_array[i+1]])
    kana_list.append(text[max(kana_index_array):])

    index = 1
    # print("kana_list",kana_list)
    for kana in kana_list[1:]:
        now_text = kana.replace("*","").replace(".","").lower()
        kana_list[index] = now_text
        if now_text == ' ':
            word_list.append('N')

        elif len(now_text)== 1:
            word_list.append(now_text.upper())
        else:
            if now_text[:-1] in i_word:
                #iIになるときはIにする
                if now_text[-1].upper() == "I":
                    word_list.append(now_text[-1].upper())
                else:
                    word_list.append('i' + now_text[-1].upper())
            elif now_text[:-1] in n_word:
                word_list.append('n' + now_text[-1].upper())
            elif now_text[0] in u_word:
                word_list.append('u' + now_text[-1].upper())
            elif now_text[:-1] in v_word:
                word_list.append(now_text[-1].upper())
            else:
                word_list.append(now_text[-1].upper())
                index += 1

        start_position = kana
    # print("word_list",word_list)

    label_index_array = get_index5(alignments,kana_index_array)

    label_index_array = (label_index_array).astype(np.int)
    label_index_array += 1

    return word_list, label_index_array

def change_judge(m_now,m_prev,m_next):
    # print("change判定")
    if m_next[-1] in "b,p,m,by,py,my":
        #論文変化3の判定
        return "N"
    elif m_prev[-1] == "A" or m_prev[-1] == "E":
        #論文変化4の判定
        return "I"
    elif m_prev[-1] == "O":
        #論文変化5の判定
        return "U"
    elif m_prev[-1] == "I":
        #論文変化6の判定
        return "I"
    elif m_prev[-1] == "U":
        return "U"
    else:
        return m_now

def get_now_path_list(length,x,y,root_path):
    out_list = []
    if os.path.exists(root_path+str(x)+'-'+str(y)):
        target_path_list = os.listdir(root_path+str(x)+'-'+str(y))
        target_path_list.sort()


        # print(str(x)+'-'+str(y))
        for point in range(length):
            out_list.append(root_path+str(x)+'-'+str(y)+'/'+target_path_list[int((len(target_path_list)-1)/length*point)])
    else:
        target_path_list = os.listdir(root_path+str(y)+'-'+str(x))
        target_path_list.sort()
        target_path_list.reverse()
        # print(str(x)+'-'+str(y))
        for point in range(length):
            out_list.append(root_path+str(y)+'-'+str(x)+'/'+target_path_list[int((len(target_path_list)-1)/length*point)])
    return out_list

def get_images_path(word_list,label_index_array,length, output_frames):
    vowels = "A,I,U,E,O,N"

    """
    for word in word_list:
        if word == "Q":
            word.replace("Q","N") #QとNは同じ処理のため、統一する
    """
    index = 0
    for word in word_list:
        if word == "Q":
            word_list[index] = "N"
        index += 1

    l_list = label_index_array.tolist()
    #print(l_list)
    path_list = []


    root_path  = './required_dataset/for_vid2vid_mouthcreate/MOUTH_small/'

    if len(word_list) != len(label_index_array):
        print("口形とラベルの長さ不一致",len(word_list) != len(label_index_array))
        len_error = True
    else:
        len_error = False
        #最初と最後に無表情を追加
        word_list = ['_'] + word_list + ['_'] + ['_']
        end_position = int((length - l_list[-1]) / 5)

        # 修正
        # l_list = [0] + l_list + [end_position]
        l_list = [0] + l_list + [l_list[-1] + end_position] + [length-1]

        # print("label_index_array",l_list)
        # print("word_list",word_list)



        #促音、撥音の特別処理
        for i in range(1, len(word_list)-2):
            m_now = word_list[i]
            m_next = word_list[i+1]

            if m_now == 'N':
                #print("nqあり")
                #print("i",i)

                #print("old_m_now",m_now)
                m_prev = word_list[i-1][-1]
                m_now = change_judge(m_now,m_prev,m_next)
                #print("new_m_now",m_now)
                word_list[i] = m_now


        #最初と最後の空白の処理
        word_list[0] = 'N'
        word_list[-2] = 'N'
        word_list[-1] = 'N'

        split_ratio = 2
        for i in range(len(word_list)-1):
            m_now = word_list[i]
            m_next = word_list[i+1]
            target_len = l_list[i+1] - l_list[i]

            if len(m_now) == 1 and len(m_next) == 1 :
                #v - v パターン
                now_list = get_now_path_list(target_len, m_now, m_next,root_path)
                #now_appearance_list, now_shape_list = get_now_aam_path_list(target_len, m_now, m_next,root_aam_path)
                #now_HP_list,now_AU_list,now_landmark_list = get_now_base_feature_path_list(target_len, m_now, m_next,root_base_feature_path)

            elif len(m_now) > 1 and len(m_next) == 1:
                #s -vパターン
                s_len = int(target_len / split_ratio)
                v_len = target_len - s_len
                siin = m_now[0].upper()
                boin = m_now[-1].upper()

                s_list = get_now_path_list(s_len, siin, boin,root_path)
                v_list = get_now_path_list(v_len, boin, m_next,root_path)
                now_list = s_list + v_list

            elif len(m_now) == 1 and len(m_next) > 1:
                #v -s パターン
                boin = m_now[-1].upper()
                siin = m_next[0].upper()

                now_list = get_now_path_list(target_len, boin, siin,root_path)

            elif len(m_now) > 1 and len(m_next) > 1:
                # s - s パターン
                siin = m_now[0].upper()
                boin = m_now[-1].upper()
                siin_2 = m_next[0].upper()

                s_len = int(target_len / split_ratio)
                #v_len = target_len - s_len*2
                s_2_len = target_len - s_len

                s_list = get_now_path_list(s_len, siin, boin,root_path)
                #v_list = get_now_path_list(v_len, boin, boin,root_path)
                s_2_list = get_now_path_list(s_2_len, boin, siin_2,root_path)

                #now_list = s_list + v_list + s_2_list
                now_list = s_list + s_2_list

            path_list.extend(now_list)


        # print(len(path_list), "==", length - 1)
    output_path_list = []
    if len_error:
        return [], True
    else:
        for p in range(output_frames):
                output_path_list.append(path_list[round((len(path_list)-1)/output_frames*p)])
        # print("frame_path_listの長さ ",len(output_path_list))
        return output_path_list, False


def noise_cut(ali,thres=0.1):

    max_index = np.argmax(ali)
    noise_array = np.where(ali<thres)[0]
    pre_last = np.where(noise_array<max_index)[0]
    if not len(pre_last) == 0:
        pre_last = pre_last[-1]
        pre_index = noise_array[pre_last]
        ali[:pre_index] = 0
    post_first = np.where(noise_array>max_index)[0]
    if not len(post_first) == 0:
        post_first = post_first[0]
        post_index = noise_array[post_first]
        ali[post_index:] = 0
    return ali

def get_index4(alignments, kana_index_array):
    ali_array = alignments.data.cpu().float().numpy()[0]
    len_kana, len_ali = kana_index_array.shape[0], ali_array.shape[0]
    scores = np.zeros((len_kana+1, len_ali+1), dtype=float)
    transition = np.zeros((len_kana+1, len_ali+1), dtype=int)
    sums = np.zeros((len_kana, len_ali+1), dtype=float)
    for i, kana in enumerate(kana_index_array):
        for j in range(len_ali):
            sums[i][j+1] = sums[i][j] + ali_array[j][kana]
    print(len_ali * len_ali * len_kana)
    for i, kana in enumerate(kana_index_array):
        for j in range(len_ali):
            for k in range(j, len_ali):
                n_score = scores[i][j]-sums[i][j]+sums[i][k+1]
                if scores[i+1][k+1] < n_score:
                    scores[i+1][k+1] = n_score
                    transition[i+1][k+1] = j
    output = []
    while len_kana:
        len_ali = transition[len_kana][len_ali]
        len_kana -= 1
        output.append(len_ali)
    return np.array(output[::-1])


#ここままだと18345みたいなのに弱い
def get_index5(alignments, kana_index_array):
    ali_array = alignments.data.cpu().float().numpy()[0]

    arg_maxes = np.array([np.argmax(ali) for ali in ali_array])
    for i in range(len(arg_maxes)):
        arg_maxes[i] = max(0, min(arg_maxes[i], ali_array.shape[0]-1))

    first_max = []
    for index in kana_index_array:
        f_max = np.where(arg_maxes == index)[0]
        if f_max.shape[0] == 0:
            f_max = [-1]
        else:
            f_max = f_max.tolist()
        first_max.append(f_max)
    first_max[0] = 0

    # 昇順になっていない場合は削除
    last_index = 0 # kana_index_arrayのindex
    last_value = 0 # last_indexでのtimesteps
    for i, f in enumerate(first_max):
        if i == 0:
            continue
        if f == [-1]:
            first_max[i] = -1
            continue
        flag = False
        for g in f:
            if g - last_value >= i - last_index:
                first_max[i] = g
                last_index = i
                last_value = g
                flag = True
                break
        if not flag:
            first_max[i] = -1
    # arg_max同士の中間をとる
    # 1, #, 5 -> 1, 3, 5
    # 3, -1, -1, 7 -> 3, 4, 5, 7
    for i, f in enumerate(first_max):
        if i == 0:
            continue
        if f == -1:
            for j, g in enumerate(first_max):
                if i >= j:
                    continue
                if g == -1:
                    continue
                for k in range(i, j):
                    first_max[k] = int(first_max[i-1] + (g - first_max[i-1]) / (j - i + 1) * (k - i + 1))
    minus1 = 0
    for f in first_max:
        if f == -1:
            minus1 += 1
    if minus1 == len(first_max) - 1:
        # これが呼ばれることはかなりないと思う
        get_index4(alignments, kana_index_array)
    else:
        if first_max[-1] == -1:
            last_index = -1
            last_value = -1
            for i, f in enumerate(first_max):
                if f == -1:
                    last_index = i-1
                    last_value = first_max[i-1]
                    break
            mean_up = int(last_value / last_index)
            for i, f in enumerate(first_max):
                if i <= last_index:
                    continue
                first_max[i] = first_max[i-1] + mean_up
    return np.array(first_max)


def get_index3(alignments, kana_index_array):
    ali_array = alignments.data.cpu().float().numpy()[0]
    timesteps = ali_array.shape[0]

    if kana_index_array[-1] >= ali_array.shape[-1]:
        print("kana_indexが長すぎるためスキップ")
        return np.array([0])

    out_box = np.array([])
    for i in range(len(kana_index_array)):
        #print(i)
        #最後のインデックスの処理
        if i == len(kana_index_array)-1:
            index = kana_index_array[i]
            now_ali = ali_array[:,index:]
            now_ali = np.sum(now_ali,axis=1)
            #plt.plot(now_ali)
            #plt.show()
            now_ali = noise_cut(now_ali,thres=0.1)
            out_box = np.r_[out_box,now_ali]
            break
        index = kana_index_array[i]
        next_index = kana_index_array[i+1]
        #print(index)
        #plt.plot(ali_array[:,index:next_index])
        now_ali = ali_array[:,index:next_index]
        now_ali = np.sum(now_ali,axis=1)
        #plt.plot(now_ali)
        #plt.show()
        now_ali = noise_cut(now_ali,thres=0.1)
        out_box = np.r_[out_box,now_ali]
    out_box = out_box.reshape((-1,timesteps))
    #print(np.argmax(out_box,axis=0))
    if not len(kana_index_array) == out_box.shape[0]:
        return np.array([0])

    else:
        #print(out_box)
        out = []
        kana_len = out_box.shape[0]
        prev_position = 0
        for k in range(kana_len):
            if not len(np.where(out_box[k]>0)[0]) == 0:
                position = np.where(out_box[k]>0)[0][0]
                #print("position:",position)
                if position<prev_position and k != kana_len-1:
                    # print("アラインメントが正常な階段でないためスキップ")
                    out = [0]
                    #out.append(position-1)
                    break
                elif position<prev_position and k == kana_len-1:
                    # print("最期だからok")
                    max_position = timesteps-1
                    estim_position = prev_position + ((max_position - prev_position) // 2)
                    out.append(estim_position)
                else:
                    if k == kana_len-1:
                        out.append(position-1)
                    else:
                        out.append(position)
                        prev_position = position
            else:
                # print("index error スキップ")
                out = [0]
                break
    return np.array(out)
