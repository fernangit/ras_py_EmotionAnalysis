# coding: utf-8
import picamera
import picamera.array
import cv2

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import random
import analyze_emotion as em

#cascade_file = "/home/pi/work/haarcascades/haarcascade_frontalface_default.xml"
cascade_file = "/home/pi/work/haarcascades/haarcascade_frontalface_alt2.xml"

emotion_model_path = './trained_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = ({0:'angry',1:'disgust',2:'fear',3:'happy',
        4:'sad',5:'surprise',6:'neutral'})
emotion_phrase =({
0:["やる気充分","熱い","格好いい"],
1:["あなたらしい","クール","個性的"],
2:["慎重","冷静","謙虚"],
3:["最高の気分","元気","明るい"],
4:["感無量","胸が一杯","涙がでそう"],
5:["びっくり","サプライズ","最高"],
6:["自然体","いい感じ","ナチュラル"]
})

def main():
    # モデルロード
    model = load_model(emotion_model_path, compile=False)
    #配列初期化
    a = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])
    predicts = np.tile(a, (10, 1))
    print(predicts)

    with picamera.PiCamera() as camera:
        with picamera.array.PiRGBArray(camera) as stream:
            camera.resolution = (320, 240)
            while True:
                # stream.arrayにRGBの順で映像データを格納
                camera.capture(stream, 'bgr', use_video_port=True)

                # グレースケールに変換
                gray = cv2.cvtColor(stream.array, cv2.COLOR_BGR2GRAY)

                # カスケードファイルを利用して顔の位置を見つける
                cascade = cv2.CascadeClassifier(cascade_file)
                face_list = cascade.detectMultiScale(gray, minSize=(100, 100), minNeighbors=3)

                for (x, y, w, h) in face_list:
#                    print("face_position:",x, y, w, h)
                    color = (0, 0, 255)
                    pen_w = 5
                    # 画像切り出し
                    dst = stream.array[y:y+h, x:x+w]
                    # 画像保存
                    cv2.imwrite("images/face.jpg", dst)
                    # 矩形表示
                    cv2.rectangle(stream.array, (x, y), (x+w, y+h), color, thickness = pen_w)
                    # 感情分析
                    predict = em.emotion(model)
                    print(predict)
                    # 平均算出
                    predicts = em.mean_emotion(predict, predicts)

                # system.arrayをウィンドウに表示
                cv2.imshow('frame', stream.array)

                # "q"でウィンドウを閉じる
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # streamをリセット
                stream.seek(0)
                stream.truncate()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()