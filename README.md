# ras_py_EmotionAnalysis
emotion analysis by python on the raspberry pi
---
* OpenCVに実装されている顔認識ライブラリを利用し、顔のランドマークを検出する。
* Kerasの学習済みモデルをTensorFlowにロードして、顔画像から感情分析を行う。

## HOW TO USE
### インストール
「https://mlbb1.blogspot.com/2018/08/keras-tensorflow.html」 を参照   
1. Kerasほか
```
sudo apt update
sudo apt install python3-sklearn python3-pil.imagetk mpg321 liblapack-dev libhdf5-dev python3-h5py
sudo pip3 install keras theano
```
2. OpenCV
```
sudo apt install python3-opencv
```
3. TensorFlow
```
sudo apt update
sudo apt install libatlas-base-dev
sudo pip3 install tensorflow
```
* MemoryErrorで終了の場合
```
sudo pip3 --no-cache-dir install tensorflow
```
4. picamera
```
sudo apt-get install python3-picamera
```
* 設定→Raspberry piの設定→インターフェース→カメラ→有効

### 環境設定
1. 感情認識学習済みモデル   
「https://github.com/oarriaga/face_classification」 から   
'fer2013_mini_XCEPTION.102-0.66.hdf5'を使用する。
2. 顔認識   
「https://github.com/opencv/opencv/tree/master/data/haarcascades」 から   
'haarcascade_frontalface_alt2.xml'を使用する。

### モジュールからの呼び出し
```
import analyze_emotion as em
emotion_model_path = './trained_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
# モデルロード
model = load_model(emotion_model_path, compile=False)
predict = em.emotion(model)

```
## 参考情報

## License
