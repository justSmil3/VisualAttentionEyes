import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.layers import *
import blocks


class LKA(keras.layers.Layer):

    def __init__(self, dim):
        super().__init__()
        self.DW = DepthwiseConv2D(kernel_size=5, padding="same")  # 2
        self.DW_D = DepthwiseConv2D(
            kernel_size=7, strides=1, padding="same", dilation_rate=3)  # 9
        self.D1x1 = Conv2D(dim, kernel_size=1, padding="same")

    def call(self, inputs):
        u = inputs
        x = self.DW(inputs)
        x = self.DW_D(x)
        x = self.D1x1(x)
        x = u * x
        return x


class Attn(keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.D1x1_1 = Conv2D(d_model, kernel_size=1, padding="same")
        self.gelu = keras.layers.Activation('gelu')
        self.LKA = LKA(d_model)
        self.D1x1_2 = Conv2D(d_model, kernel_size=1, padding="same")

    def call(self, inputs):
        u = inputs
        x = self.D1x1_1(inputs)
        x = self.gelu(x)
        x = self.LKA(x)
        x = x + u
        return x


class FFN(keras.layers.Layer):
    def __init__(self, hidden_features=None, out_features=None):
        super().__init__()
        self.D1x1_1 = Conv2D(hidden_features, kernel_size=1)
        self.DW = DepthwiseConv2D(kernel_size=3, padding="same")
        self.act = keras.layers.Activation('gelu')
        self.D1x1_2 = Conv2D(out_features, kernel_size=1, padding="same")
        self.Dropout = Dropout(.5)

    def call(self, inputs):
        x = self.D1x1_1(inputs)
        x = self.DW(x)
        x = self.act(x)
        x = self.Dropout(x)
        x = self.D1x1_2(x)
        x = self.Dropout(x)

        return x


class Stage(Layer):
    def __init__(self, dim, mlp_ratio=4, drop=0., drop_path=0., act_layer=keras.layers.Activation('gelu')):
        super().__init__()
        self.bn1 = BatchNormalization()
        self.attn = Attn(dim)
        self.drop_path = blocks.DropPath(
            drop_path) if drop_path > 0. else tf.identity

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.bn2 = BatchNormalization()
        self.ffn = FFN(mlp_hidden_dim, dim)

        layerscale_initvalue = 1e-2
        self.layerscale_1 = tf.Variable(
            layerscale_initvalue * tf.ones((dim))
        )

        self.layerscale_2 = tf.Variable(
            layerscale_initvalue * tf.ones((dim))
        )

    def call(self, inputs):
        print(inputs.shape)
        x = inputs + self.drop_path(tf.expand_dims(tf.expand_dims(tf.expand_dims(
            self.layerscale_1, [0]), [0]), [0]) * self.attn(self.bn1(inputs)))

        x = x + self.drop_path(tf.expand_dims(tf.expand_dims(tf.expand_dims(
            self.layerscale_2, [0]), [0]), [0]) * self.ffn(self.bn2(x)))
        return x


class OverlapPatchEmbed(keras.layers.Layer):

    def __init__(self, patch_size=7, stride=4, embed_dim=768):
        super().__init__()
        patch_size = blocks.to_2tuple(patch_size)
        self.proj = Conv2D(embed_dim, patch_size, strides=stride,
                           padding="same")  # (patch_size[0] // 2, patch_size[1]//2)
        self.norm = BatchNormalization()

    def call(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, W, H


class VAN(keras.models.Model):
    def __init__(self, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=None,
                 depths=[3, 4, 6, 3], num_stages=4):
        super().__init__()
        if norm_layer == None:
            norm_layer = keras.layers.LayerNormalization()
        self.depths = depths
        self.num_stages = num_stages
        # dpr = tf.linspace(0, drop_path_rate, sum(depths))
        dpr = np.linspace(0, drop_path_rate, sum(depths))
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            embed_dim=embed_dims[i])
            stage = [Stage(embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate,
                           drop_path=dpr[cur + j]) for j in range(depths[i])]
            norm = keras.layers.LayerNormalization(axis=-1)
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"stage{i + 1}", stage)
            setattr(self, f"norm{i + 1}", norm)

        self.head = Dense(embed_dims[3], activation="sigmoid")
        self.classifyer = Dense(1, activation="linear")

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"stage{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)

            for blk in block:
                x = blk(x)

            x = tf.reshape(x, [x.shape[0], x.shape[1], -1]
                           )
            x = tf.transpose(x, perm=[0, 2, 1])
            x = norm(x)
            if i != self.num_stages - 1:
                x = tf.reshape(x, [B, H, W, -1])
                x = tf.transpose(x, [0, 3, 1, 2])

        return tf.reduce_mean(x, axis=1)

    def call(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = self.classifyer(x)
        return x
