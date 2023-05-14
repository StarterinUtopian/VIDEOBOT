import numpy as np
import os

def plus_blink(res, at, blink_type, need_frame, blink):
    if len(res) - at < need_frame:
        return res
    for i in range(need_frame):
        res[i+at][105:140, 70:190, :] = blink[blink_type][i][105:140, 170:290, :]
    return res

def get_imgs_points(output_path_list):
    imgs = []
    points = []
    for path in output_path_list:
        if os.path.exists(path):# and os.path.exists(path.replace('MOUTH_small_revised_edge_512','MOUTH_small_revised_npy_512').replace('png','npy')):
            points.append(np.load(path.replace('jpg','npy').replace('imgs','npy')))#.replace('MOUTH_small_revised_edge_512','MOUTH_small_revised_npy_512').))
        else:
            pass
    return points

def get_points(output_path_list, start_blink):
    if start_blink == -1:
        return
    points = []
    for i, id in enumerate(output_path_list):
        if i > (start_blink - 3) and i < (start_blink + 5):
            points.append(all_point[id])
        else:
            points.append([])
    return points

def delete_between(texts, list_1, list_2):
    cnt = 0
    res = ""
    for s in texts:
        if s in list_1:
            cnt += 1
        if s in list_2:
            cnt += -1
            continue
        if cnt == 0:
            res += s
    return res

def add_blink(points):
    scale = [0.2, 0.8, 1, 0.6, 0.4]
    points_1 = points[0]
    points_2 = points[1]
    points_3 = points[2]
    points_4 = points[3]
    points_5 = points[4]

    distance_l = (points_3[40:42] - points_3[37:39])*(1.01,1)
    distance_r = (points_3[46:48] - points_3[43:45])*(1,1.01)

    points_1[37:39] = points_1[37:39] + distance_l * scale[0]
    points_1[43:45] = points_1[43:45] + distance_r * scale[0]
    points_2[37:39] = points_2[37:39] + distance_l * scale[1]
    points_2[43:45] = points_2[43:45] + distance_r * scale[1]
    if scale[2] == 1:
        points_3[37:39] = points_3[40:42]
        points_3[43:45] = points_3[46:48]
    else:
        points_3[37:39] = points_3[37:39] + distance_l * scale[2]
        points_3[43:45] = points_3[43:45] + distance_r * scale[2]
    points_4[37:39] = points_4[37:39] + distance_l * scale[3]
    points_4[43:45] = points_4[43:45] + distance_r * scale[3]
    points_5[37:39] = points_5[37:39] + distance_l * scale[4]
    points_5[43:45] = points_5[43:45] + distance_r * scale[4]

    return [points_1, points_2, points_3, points_4, points_5]

def is_num(s):
    try:
        float(s.replace("。","."))
    except ValueError:
        return False
    else:
        return True