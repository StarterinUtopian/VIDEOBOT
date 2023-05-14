export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/usr/local/lib64:/usr/lib64
g++ -std=c++11 add_subtitling.cpp -I/usr/local/include/opencv4/ -I/home/rx-0/Qt5.14.1/5.14.1/android/include -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -lopencv_imgproc -lQt5Gui -lQt5Core -lGL
./a.out
rm -f final_result.mp4
ffmpeg -i merge.mp4 -i ../.temp_output.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -strict -2 final_result.mp4