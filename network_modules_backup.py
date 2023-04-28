import tensorflow as tf
import numpy as np
import keras.layers
from keras.layers import *
import keras.models
from keras.activations import gelu
import blocks


class LKA(keras.layers.Layer):

    def __init__(self, dim):
        super().__init__()
        self.DW = DepthwiseConv2D(kernel_size=5, padding=2)
        self.DW_D = DepthwiseConv2D(
            kernel_size=7, strides=1, padding=9, dilation_rate=3)
        self.D1x1 = Conv2D(dim, kernel_size=1)

    def call(self, inputs):
        u = inputs.clone()
        x = self.DW(inputs)
        x = self.DW_D(x)
        x = self.D1x1(x)
        x = u * x
        return x


class Attn(keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.D1x1_1 = Conv2D(d_model, kernel_size=1)
        self.gelu = keras.layers.Activation('gelu')
        self.LKA = LKA(d_model)
        self.D1x1_2 = Conv2D(d_model, kernel_size=1)

    def call(self, inputs):
        u = inputs.copy()
        x = self.D1x1_1(inputs)
        x = self.gelu(x)
        x = self.LKA(x)
        x = x + u
        return x


class FFN(keras.layers.Layer):
    def __init__(self, hidden_features=None, out_features=None):
        super().__init()
        self.D1x1_1 = Conv2D(hidden_features, Kernal_size=1)
        self.DW = DepthwiseConv2D(kernel_size=3)
        self.act = keras.layers.Activation('gelu')
        self.D1x1_2 = Conv2D(out_features, Kernal_size=1)
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
        self.layerscale_1 = tf.variable(
            layerscale_initvalue * tf.ones((dim))
        )

        self.layerscale_2 = tf.variable(
            layerscale_initvalue * tf.ones((dim))
        )

    def call(self, inputs):
        x = inputs + self.drop_path(tf.expand_dims(tf.expand_dims(
            self.layerscale_1, [-1]), [-1]) * self.attn(self.bn1(inputs)))
        x = x + self.drop_path(tf.expand_dims(tf.expand_dims(
            self.layerscale_2, [-1]), [-1]) * self.ffn(self.bn2(x)))
        return x


class OverlapPatchEmbed(keras.layers.Layer):

    def __init__(self, patch_size=7, stride=4, embed_dim=768):
        super().__init__()
        patch_size = blocks.to_2tuple(patch_size)
        self.proj = Conv2D(embed_dim, patch_size, stride=stride,
                           padding=(patch_size[0] // 2, patch_size[1]//2))
        self.norm = BatchNormalization()

    def call(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, W, H


class VAN(keras.models.Model):
    def __init__(self, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], num_stages=4, flag=False):
        super().__init__()
        if flag == False:
            self.num_classes = num_classes
        self.depths = depths

        dpr = [x.item() for x in tf.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i+1],
                                            embed_dim=embed_dims(i))
            stage = tf.ModuelList([Stage(
                embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

            self.head = Dense(embed_dims[3], activation="identity")
