# ras_py_EmotionAnalysis
emotion analysis by python on the raspberry pi
---
* OpenCVに実装されている顔認識ライブラリを利用し、顔のランドマークを検出する。   
* Kerasの学習済みモデルをTensorFlowにロードして、顔画像から感情分析を行う。   

## HOW TO USE
### インストール
1. TensorFlow  
```
$ git clone https://github.com/you0708/adrsir.git
$ cd adrsir
```  
2. Keras
```
$ git clone https://github.com/you0708/adrsir.git
$ cd adrsir
```  
### 環境設定
1. OpenCV  
2. TensorFlow  
3. Keras  

### API  

#### モジュールからの呼び出し
```
import tplink_smartplug_py3 as plug
plug.control('192.168.0.2', 'on')
```
