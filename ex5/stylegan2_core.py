import numpy as np
import tensorflow as tf

LATENT_DIM = 256
EPS = 1e-8


class MappingNetwork(tf.keras.Model):
    def __init__(self, latent_dim=LATENT_DIM, num_layers=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.layers_list = [
            tf.keras.layers.Dense(latent_dim) for _ in range(num_layers)
        ]
        self.act = tf.keras.layers.LeakyReLU(0.2)

    def call(self, z):
        x = z / (tf.norm(z, axis=-1, keepdims=True) + EPS)
        for layer in self.layers_list:
            x = self.act(layer(x))
        return x


class ModulatedConv2d(tf.keras.layers.Layer):
    def __init__(self, in_ch, out_ch, kernel_size=3, demodulate=True, upsample=False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.style_proj = tf.keras.layers.Dense(in_ch, bias_initializer="ones")
        w_init = tf.random_normal_initializer(stddev=1.0 / np.sqrt(in_ch * kernel_size * kernel_size))
        self.kernel = tf.Variable(
            w_init(shape=(kernel_size, kernel_size, in_ch, out_ch), dtype=tf.float32),
            trainable=True, name="mod_conv_kernel")

    def call(self, x, w):
        batch_size = tf.shape(x)[0]
        if self.upsample:
            h, wd = tf.shape(x)[1], tf.shape(x)[2]
            x = tf.image.resize(x, (h * 2, wd * 2), method="nearest")

        s = self.style_proj(w)  # (B, in_ch)
        kernel = self.kernel[None, ...]  # (1, k, k, in_ch, out_ch)
        s_b = s[:, None, None, :, None]  # (B, 1, 1, in_ch, 1)
        w_hat = kernel * s_b  # (B, k, k, in_ch, out_ch)

        if self.demodulate:
            demod = tf.math.rsqrt(tf.reduce_sum(tf.square(w_hat), axis=[1, 2, 3], keepdims=True) + EPS)
            w_hat = w_hat * demod

        def conv_one(inputs):
            xi, ki = inputs
            return tf.nn.conv2d(xi[None, ...], ki, strides=1, padding="SAME")[0]

        out = tf.map_fn(conv_one, (x, w_hat), fn_output_signature=tf.float32)
        return out


class NoiseInjection(tf.keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.scale = tf.Variable(tf.zeros((1, 1, 1, channels)), trainable=True, name="noise_scale")

    def call(self, x):
        noise = tf.random.normal(shape=(tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1))
        return x + self.scale * noise


class StyleGAN2Block(tf.keras.layers.Layer):
    def __init__(self, in_ch, out_ch, upsample=True):
        super().__init__()
        self.mod_conv = ModulatedConv2d(in_ch, out_ch, upsample=upsample)
        self.noise = NoiseInjection(out_ch)
        self.bias = tf.Variable(tf.zeros((1, 1, 1, out_ch)), trainable=True)
        self.act = tf.keras.layers.LeakyReLU(0.2)
        self.to_rgb = ModulatedConv2d(out_ch, 1, kernel_size=1, demodulate=False)

    def call(self, x, w):
        x = self.mod_conv(x, w)
        x = self.noise(x)
        x = self.act(x + self.bias)
        rgb = self.to_rgb(x, w)
        return x, rgb


class StyleGAN2Generator(tf.keras.Model):
    def __init__(self, latent_dim=LATENT_DIM, channels=(256, 128, 64, 32), out_res=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.mapping = MappingNetwork(latent_dim)
        self.const_input = tf.Variable(
            tf.random.normal((1, 4, 4, channels[0])), trainable=True, name="const_input")
        self.initial_to_rgb = ModulatedConv2d(channels[0], 1, kernel_size=1, demodulate=False)
        self.blocks = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            self.blocks.append(StyleGAN2Block(in_ch, out_ch, upsample=True))
            in_ch = out_ch
        self.n_blocks = len(self.blocks)

    def call(self, z, w_mix=None, mix_at=None):
        batch_size = tf.shape(z)[0]
        w = self.mapping(z)
        if w_mix is not None:
            w2 = self.mapping(w_mix)

        x = tf.tile(self.const_input, [batch_size, 1, 1, 1])
        rgbs = [self.initial_to_rgb(x, w)]  # 4x4 stage
        for i, block in enumerate(self.blocks):
            cur_w = w2 if (w_mix is not None and mix_at is not None and i >= mix_at) else w
            x, rgb = block(x, cur_w)
            rgbs.append(rgb)
        return rgbs  # list of RGB outputs at each resolution, last = final image


def style_mixing_demo(generator, batch_size=4, mix_at=2):
    z1 = tf.random.normal((batch_size, generator.latent_dim))
    z2 = tf.random.normal((batch_size, generator.latent_dim))
    rgbs = generator(z1, w_mix=z2, mix_at=mix_at)
    return rgbs[-1]


if __name__ == "__main__":
    gen = StyleGAN2Generator()
    z = tf.random.normal((2, LATENT_DIM))
    outputs = gen(z)
    for r in outputs:
        print(r.shape)
    mixed = style_mixing_demo(gen)
    print("style-mixed output:", mixed.shape)
