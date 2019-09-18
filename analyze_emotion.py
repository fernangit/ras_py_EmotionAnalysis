from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import random

rgb_image = './images/face.jpg'
#rgb_image = './images/mika_74.jpg'
#rgb_image = './images/mika_110.jpg'
#rgb_image = './images/rika_67.jpg'
#rgb_image = './images/rika_132.jpg'

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

def init_emotion(n):
    # モデルロード
    model = load_model(emotion_model_path, compile=False)
    #配列初期化
    a = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])
    predicts = np.tile(a, (n, 1))

    return model, predicts

#感情予測
def emotion(model):
    load_img = image.load_img(rgb_image, grayscale=True , target_size=(64, 64))
    x = image.img_to_array(load_img)
    x = np.expand_dims(x, axis=0) / 255

    predict = model.predict(x)
    print(predict)
    emotion_label_arg = np.argmax(predict)
    emotion_text = emotion_labels[emotion_label_arg]
    emotion_ratio = np.max(predict)
    emotion_phrase_rand = random.choice(emotion_phrase[emotion_label_arg])
    print(emotion_text,emotion_ratio)
    print('{}ですね。'.format(emotion_phrase_rand))

    return predict

# 平均化
def mean_emotion(predict, predicts):
    #結果挿入
    predicts = np.vstack((predicts, predict))
    predicts = np.delete(predicts, 0, axis=0)
#    print(predicts)

    #平均算出
    predict_mean = np.mean(predicts, axis=0)
#    print(predict_mean)
#    print('shape:', predict_mean.shape)
#    print('rank:', predict_mean.ndim) 

    for i in range(len(predict_mean)):
        print(emotion_labels[i], predict_mean[i])

    emotion_label_arg = np.argmax(predict_mean)
    emotion_text = emotion_labels[emotion_label_arg]
    emotion_ratio = np.max(predict_mean)
    emotion_phrase_rand = random.choice(emotion_phrase[emotion_label_arg])
#    print(emotion_text,emotion_ratio)
    print('{}'.format(emotion_phrase_rand))

    return predicts

def main():
    # 初期化
    model, predicts = init_emotion(10)
    print(predicts)

    #感情分析
    predict = emotion(model)

    #平均算出
    predicts = mean_emotion(predict, predicts)

if __name__ == '__main__':
    main()