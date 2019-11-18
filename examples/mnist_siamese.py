# %%

'''Trains a Siamese MLP on pairs of digits from the MNIST dataset.

It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for more details).

# References

- Dimensionality Reduction by Learning an Invariant Mapping
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Gets to 97.2% test accuracy after 20 epochs.
2 seconds per epoch on a Titan X Maxwell GPU
'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import plot_model
import numpy as np

num_classes = 10
epochs = 20

# %%
A = np.asarray([
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ],
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90]
        ],
        [
            [100, 200, 300],
            [400, 500, 600],
            [700, 800, 900]
        ]
    ])

B = np.asarray(
        [9, 8, 7]
    )

A = K.variable(A)
B = K.variable(B)
# %%
# K.get_value(A)

# K.sum(K.square(A - B), axis=0, keepdims=True)
print(K.get_value(K.sum(A,keepdims=True,axis=0)))
print(K.get_value(K.sum(A,keepdims=True,axis=1)))
print(K.get_value(K.sum(A,keepdims=True,axis=2)))
# print(K.get_value(K.sum(A,keepdims=True)))

# %%


def euclidean_distance(vects):
    '''
    この関数では サンプル間の距離を計算している
    ユークリッド距離
    '''
    x, y = vects
    # 入力は隠れそうを通るので2つのサンプルになるので
    # (None, 128) が入力される
    # そのため axis=1 を指定している
    # 因みに None になっている理由はサンプル数を可変にするため値が決まっていないとするため
    # https://arakan-pgm-ai.hatenablog.com/entry/2017/05/06/214113

    # sum で axis を指定すると指定した軸の次元が1になる
    # (3, 3, 3) のデータがあるとすると
    # axis=0 -> (1, 3, 3)
    # axis=1 -> (3, 1, 3)
    # axis=2 -> (3, 3, 1)
    # となる

    # ここでは (None, 128) であったので (None, 1) になる
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    # K.epsilon() は計算で使用される微小量を返す
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    
    Positive と negative のペアを作成する
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            # z1, z2 はインデックス
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            # x は画像データ
            # ここでポジティブペアを作成する
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            # ここでネガティブペアを作成している
            pairs += [[x[z1], x[z2]]]
            # ポジティブペアを 1, ネガティブペアを 0 にラベリングしている
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


# the data, split between train and test sets
## MNIST 手書き数字データベースをロード
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
## 60,000枚の28x28の白黒データなので imput_shape == [28, 28]
input_shape = x_train.shape[1:]

# %%
# create training+test positive and negative pairs
## このリスト内包表現でインデックスに対応した y_train のデータのインデックスを格納した配列を取得する
## y_train が [5 0 4 ... 5 6 8]のようなデータの時
## digit_indices は
## [[1 ...], # 0
## ...
## [0 ... N-2], # 5
## ...
## [...]] # 9
## のような二次元配列が取得できる
digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
te_pairs, te_y = create_pairs(x_test, digit_indices)

# %%
print(tr_pairs[0])
print(tr_y[0])

# %%
# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

## euclidean_distance はサンプル間の距離を計測する関数
# distance = Lambda(euclidean_distance,
#                   output_shape=eucl_dist_output_shape)([processed_a, processed_b])
distance = Lambda(euclidean_distance)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))


# %%
plot_model(model, to_file='model.png', show_shapes=True)