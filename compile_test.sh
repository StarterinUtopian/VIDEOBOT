export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/usr/local/lib64:/usr/lib64
g++ -std=c++11 hoge.cpp -I/usr/local/include/opencv4/ -I/home/rx-0/Qt5.14.1/5.14.1/android/include -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -lopencv_imgproc -lQt5Gui -lQt5Core -lGL
