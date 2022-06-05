# Ballet Posture Coaching Program🩰

## How to start

- clone this repository

- download zip file at https://github.com/CMU-Perceptual-Computing-Lab/openpose and unzip openpose-master.zip

- execute getModels.sh

- download "pose_iter_160000.caffemodel" at
  http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel
  and put it model directory

- download "pose_iter_160000.caffemodel" at
  http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel
  and put it model directory

- execute main.py

## 사용 방법

- **Select Mode**
  - 실시간 코칭을 원한다면 1️⃣을 눌러 Realtime Mode 선택
  - 10초 타이머를 설정하여 원하는 장면을 찍어 코칭받고 싶다면 2️⃣를 눌러 Detailed Mode 선택
- **다시 찍기**
  - Detailed Mode 선택시 화면에 10초 카운트 다운이 표시되며 사진이 찍히고 결과를 확인할 수 있다. 다시 찍고 싶으면 키보드 q키를 누르면 된다.
- **종료**
  - 키보드 ESC 누르기

## crawling
- move to crawling directory and create input directory
- down load chromedriver_win32 and put it crawling directory
- execute crawling.py

## 개발환경

vscode, colab
