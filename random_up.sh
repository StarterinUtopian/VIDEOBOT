#!/bin/sh

min=`expr $RANDOM % 1800` # 0～30分 のいずれか
echo $min
# sleep $min
cd ~/videobot_YouTubevers/57_Youtube_VideoBot/Youtube; 
python send2youtube.py --file="test2.mp4"