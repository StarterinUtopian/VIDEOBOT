# -*- coding: utf-8 -*-
import numpy as np
import cv2
import sys
import os
import random
import copy
from tqdm import tqdm

from PIL import Image, ImageDraw, ImageFont

font_size = 25
size_face = 512
img_h = 720
img_w = 1280
fontPIL = "/home/elleair/videobot_YouTubevers/57_Youtube_VideoBot/add_subtitling/DFLgs9.ttc"

fontFace1, fontScale1, color1 = fontPIL, font_size, (255, 255, 255)
fontFace2, fontScale2, color2 = fontPIL, font_size, (220, 220, 220)
fontFace3, fontScale3, color3 = fontPIL, int(font_size*1.5), (255, 255, 255)

def die(s):
    print(s)
    sys.exit()

def pil2cv(imgPIL):
    imgCV_RGB = np.array(imgPIL, dtype = np.uint8)
    imgCV_BGR = np.array(imgPIL)[:, :, ::-1]
    return imgCV_BGR

def cv2pil(imgCV):
    # imgCV_RGB = imgCV[:, :, ::-1]
    # imgCV[:, :, (0,2)] = imgCV[:, :, (2, 0)]
    # print(type(imgCV_RGB))
    # print(f"size of imgCV_RGB: {imgCV_RGB.shape}")
    imgPIL = Image.fromarray(imgCV, mode="RGB")
    # print(f"type of imgPIL {type(imgPIL)}")
    return imgPIL

def cv2_putText(img, text, org, fontFace, fontScale, color):
    # print("called cv2_putText")
    x, y = org
    b, g, r = color
    colorRGB = (r,g,b)
    # print(f"size of img(2): {img.shape}")
    imgPIL = cv2pil(img)
    # imgPIL.save("test1.png")
    # imgPIL = Image.fromarray(img, mode="RGB")
    draw = ImageDraw.Draw(imgPIL)

    fontPIL = ImageFont.truetype(font = fontFace, size = fontScale)
    # w, h = draw.textsize(text, font = fontPIL)
    # draw.text(xy = (x+600,y-h), text = text, fill = colorRGB, font = fontPIL)
    draw.text(xy = (x,y), text = text, fill = colorRGB, font = fontPIL)
    # imgPIL.save("test2.png")
    imgCV = pil2cv(imgPIL)
    # cv2.imwrite("test3.png", imgCV)
    return imgCV

# 画像に文字を入れる関数
def putTextJpn(title, img, lines, n, max_text, current_frame, max_frames):
    w, h, color = img.shape
    font_size_ = font_size * 11 // 8
    max_words = w // font_size_

    sentence = (len(current_sentence) + max_words - 1) // max_words

    alr = 0
    # last_sentence_up = sentence + 1
    last_sentence_up = 0
    # last_sentence_down = 1 - sentence
    last_sentence_down = 0
    gap = sentence + 1
    # if n != max_text:
    #     gap = ((len(lines[n+1]) + max_words - 1) // max_words + 1)
    # 読んでる部分
    for i in range(sentence):
        need_len = int(len(current_sentence) / sentence + (i < (len(current_sentence) % sentence)))
        read_str = current_sentence[alr:alr+need_len]
        alr += need_len
        need_space = w * len(read_str) / max_words
        _w = (w - need_space) / 2
        _w+=600
        # _h = w*3/4 - (font_size_) * 2*(sentence - i)-gap*font_size_*(1.0 *current_frame/max_frames)
        _h = w*3/4 + (font_size_)*i-gap*font_size_*(1.0 *current_frame/max_frames)
        if current_frame==max_frames-1 or current_frame==0:
            print(f"真ん中need_len:{need_len}")
            print("真ん中：read_str",read_str)
            print("\h",_h)
            print("i-gap*(1.0 *current_frame/max_frames): "+str(i)+str(-gap*(1.0 *current_frame/max_frames)))
        img = cv2_putText(img, read_str, (_w, _h), fontFace1, fontScale1, color1)
        img = cv2_putText(img, title, (600, 70), fontFace3, fontScale3, color3)


    # 読んでる部分より上
    for n_ in reversed(range(n)):
        line = lines[n_]
        if current_frame==max_frames-1 or current_frame==0:
            print(line)
        sentence_ = (len(line) + max_words - 1) // max_words
        alr = 0
        # if sentence_ >= 2 and lines[n_+1] == current_sentence:
        #     last_sentence_up -= sentence_ - 1
        for i in range(sentence_):
            need_len = len(line) / sentence_ + (i < (len(line) & sentence_))
            need_len = int(need_len)
            read_str = line[alr:alr+need_len]
            alr += need_len
            need_space = w * len(read_str) / max_words
            _w = (w - need_space) / 2
            _w+=600
            # _h = w*2/3 - (font_size_) * (last_sentence_up + sentence - (sentence - sentence_) - i)-gap*font_size_*(1.0 *current_frame/max_frames)
            _h = w*3/4 - (font_size_) * (sentence_ + 1 - i + last_sentence_up)-gap*font_size_*(1.0 *current_frame/max_frames)
            if current_frame==max_frames-1 or current_frame==0:
                print(f"上need_len:{need_len}")
                print("上：read_str",read_str)
                print("\h",_h)
                print("(2*(sentence_ - i) + last_sentence_up)-gap*(1.0 *current_frame/max_frames): "+str(-(2*sentence_ - i + last_sentence_up))+str(-gap*(1.0 *current_frame/max_frames)))
            img = cv2_putText(img, read_str, (_w, _h), fontFace2, fontScale2, color2)
            img = cv2_putText(img, title, (600, 70), fontFace3, fontScale3, color3)
        last_sentence_up += sentence_ +1 
    if n <= max_text:
        max_down = max_text - n

    # 読んでる部分より下
    for n_ in range(n+1, n+max_down+1):
        line = lines[n_]
        sentence_ = (len(line) + max_words - 1) // max_words
        alr = 0
        # if sentence >= 2 and lines[n_-1] == current_sentence:
        #     last_sentence_down -= sentence - 1
        for i in range(sentence_):
            if (i < len(line) % sentence_):
                additional = 1
            else:
                additional = 0
            need_len = len(line) // sentence_ + additional
            read_str = line[alr: alr+need_len]
            alr += need_len
            need_space = w * len(read_str)  // max_words
            _w = (w - need_space) / 2
            _w+=600
            # _h = w*2/3 + (font_size_) * (last_sentence_down + sentence + i)-(gap)*font_size_*(1.0 *current_frame/max_frames)
            _h = w*3/4 + (font_size_) * (last_sentence_down + 1 + sentence + i)-(gap)*font_size_*(1.0 *current_frame/max_frames)
            if current_frame==max_frames-1 or current_frame==0:
                print(f"下：need_len:{need_len}")
                print("下：read_str",read_str)
                print("\h",_h)
                print("(last_sentence_down + 2*(sentence + i))-gap*(1.0 *current_frame/max_frames): "+str(last_sentence_down + 2*(sentence + i))+str(-gap*(1.0 *current_frame/max_frames)))
            img = cv2_putText(img, read_str, (_w, _h), fontFace2, fontScale2, color2)
            img = cv2_putText(img, title, (600, 70), fontFace3, fontScale3, color3)
        last_sentence_down += sentence_ + 1
    return  img

def errorInStr(s):
    return "error" in s or "Error" in s


# int main(int argc, char* argv[]){
if __name__ == '__main__':
  print("called")
  filename_video = sys.argv[2]
  result_txt_path = "./"+filename_video+"_result.txt"
  if not os.path.exists(result_txt_path):
      die("result.txt 読み取り失敗")
  with open(result_txt_path) as f:
      lines = f.readlines()
  title = sys.argv[1].replace('_', ' ')
  print("at add_subtitling result_txt_path is", result_txt_path)
  # print(lines)


  original_lines = []
  image_folders = []
  image_num = []
  wave_pathes = []

  for cnt, line in enumerate(lines):
      if cnt % 3 == 0:
          original_lines.append(line)
      elif cnt % 3 == 1:
          datas = line.split(" ")
          # print(datas)
          tmp1 = datas[0]
          tmp2 = datas[1]
          tmp3 = datas[2]

          tmp2_num = int(tmp2)
          image_folders.append(tmp1)
          image_num.append(tmp2_num)
          wave_pathes.append(tmp3)


  total_frames, real_add = 0, 0
  total_frames = sum(image_num)

  fps = 24.

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  writer = cv2.VideoWriter("merge_" + filename_video + ".mp4", fourcc, fps, (img_w,img_h))

  movie_num = len(original_lines)
  all_cnt = 0
  padding_img = 50
  # 一文ごとに対して、この処理を行う。
  print("at add_subtitling.py movie_num", movie_num)
  for i in tqdm(range(movie_num)):
      current_sentence = original_lines[i]
      num = ""
      for j in reversed(range(len(image_folders[i]))):
          if image_folders[i][j] == '_':
              break
      num += image_folders[i][j]
      rand_folder = random.randint(0, 19)
      i_num = str(i)

      background_path = "./tmp_data/background_images/" + filename_video + "/" + i_num + ".jpg"

      if not os.path.exists(background_path):
          background_path = "./required_dataset/crystalmethod.jpg"

      background_ = cv2.imread(background_path)
      if background_ is None:
          background_path = "./required_dataset/crystalmethod.jpg"
          background_ = cv2.imread(background_path);

      # background = np.zeros((img_w, img_h), dtype=np.uint8)
      background_ = cv2.resize(background_, (img_w, img_h))
      # print(f"背景の配列:{background_.shape}")

      all_cnt += image_num[i]
      add_sub = True

      if original_lines[i] == "。" or errorInStr(image_folders[i]):
          add_sub = 0

      # １フレームごとにこの処理をする。
      for j in tqdm(range(image_num[i])):
          background_ = cv2.imread(background_path)
          background_ = cv2.resize(background_, (img_w, img_h))
          img_path = image_folders[i] + "/" + str(j) + ".jpg"
          # 顔写真読み込み
          img = cv2.imread(img_path)
          # 顔写真を縮小する
          img = cv2.resize(img, (size_face, size_face))
          # 背景の左上に顔写真を配置
          background_[0:size_face, 0:size_face] = img
          # 背景の一部を暗くする
          background_[:,550:1150,] = background_[:,550:1150,]*0.8

          if add_sub:
              # テキスト配置
              img = putTextJpn(title, background_, original_lines, i, movie_num-1, j, image_num[i])

          complete_img = np.uint8(img)
          # フレーム書き込み
          writer.write(complete_img)
          # writer.write(background_)
          real_add += 1

  writer.release()
