# coding: utf-8
import picamera
import picamera.array
import cv2
import analyze_emotion as em

#cascade_file = "/home/pi/work/haarcascades/haarcascade_frontalface_default.xml"
cascade_file = "/home/pi/work/haarcascades/haarcascade_frontalface_alt2.xml"

def face_analyze()
    # 初期化
    model, predicts = em.init_emotion(10)
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

def main():
    # カメラから顔画像取り込み、表情分析
    face_analyze()

if __name__ == '__main__':
    main()