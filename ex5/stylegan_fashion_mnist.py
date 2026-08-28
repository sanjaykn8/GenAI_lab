import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

LATENT_DIM = 256
IMG_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 20
LR = 2e-4
D_REG = 16
GAMMA = 10.0
OUT_DIR = "./outputs_stylegan_fmnist"
os.makedirs(OUT_DIR, exist_ok=True)
EPS = 1e-8

tf.random.set_seed(0)
np.random.seed(0)


def load_fashion_mnist():
    (x_train, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32")
    x_train = np.expand_dims(x_train, -1)
    x_train = tf.image.resize(x_train, (IMG_SIZE, IMG_SIZE)).numpy()
    x_train = (x_train / 127.5) - 1.0
    return x_train


def make_dataset(images, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices(images)
    return ds.shuffle(len(images)).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


class MappingNetwork(tf.keras.Model):
    def __init__(self, latent_dim=LATENT_DIM, num_layers=8):
        super().__init__()
        self.layers_list = [tf.keras.layers.Dense(latent_dim) for _ in range(num_layers)]
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
            w_init(shape=(kernel_size, kernel_size, in_ch, out_ch), dtype=tf.float32), trainable=True)

    def call(self, x, w):
        if self.upsample:
            h, wd = tf.shape(x)[1], tf.shape(x)[2]
            x = tf.image.resize(x, (h * 2, wd * 2), method="nearest")

        batch_size = tf.shape(x)[0]
        h, wd = tf.shape(x)[1], tf.shape(x)[2]

        s = self.style_proj(w)
        w_hat = self.kernel[None, ...] * s[:, None, None, :, None]  # (B, k, k, in, out)
        if self.demodulate:
            demod = tf.math.rsqrt(tf.reduce_sum(tf.square(w_hat), axis=[1, 2, 3], keepdims=True) + EPS)
            w_hat = w_hat * demod

        # Vectorized per-sample conv via a single grouped convolution instead of a
        # Python-level tf.map_fn loop over the batch: fold the batch dim into the
        # channel dim on both the input and the kernel, then run ONE conv2d call
        # with groups == batch_size. Same math as looping per-sample, but runs as
        # one fused op instead of `batch_size` separate eager/graph conv calls.
        x_grouped = tf.transpose(x, [1, 2, 0, 3])                       # (H, W, B, Cin)
        x_grouped = tf.reshape(x_grouped, [1, h, wd, batch_size * self.in_ch])

        w_grouped = tf.transpose(w_hat, [1, 2, 3, 0, 4])                # (k, k, Cin, B, Cout)
        w_grouped = tf.reshape(w_grouped, [self.k, self.k, self.in_ch, batch_size * self.out_ch])

        out = tf.nn.conv2d(x_grouped, w_grouped, strides=1, padding="SAME")
        out = tf.reshape(out, [h, wd, batch_size, self.out_ch])
        out = tf.transpose(out, [2, 0, 1, 3])                           # (B, H, W, Cout)
        return out


class NoiseInjection(tf.keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.scale = tf.Variable(tf.zeros((1, 1, 1, channels)), trainable=True)

    def call(self, x):
        noise = tf.random.normal((tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1))
        return x + self.scale * noise


class GenBlock(tf.keras.layers.Layer):
    def __init__(self, in_ch, out_ch, upsample=True):
        super().__init__()
        self.mod_conv = ModulatedConv2d(in_ch, out_ch, upsample=upsample)
        self.noise = NoiseInjection(out_ch)
        self.bias = tf.Variable(tf.zeros((1, 1, 1, out_ch)), trainable=True)
        self.act = tf.keras.layers.LeakyReLU(0.2)

    def call(self, x, w):
        x = self.mod_conv(x, w)
        x = self.noise(x)
        return self.act(x + self.bias)


class Generator(tf.keras.Model):
    """Lightweight StyleGAN2-style generator, single-channel output, 4x4 -> 32x32."""

    def __init__(self, latent_dim=LATENT_DIM, channels=(256, 128, 64, 32)):
        super().__init__()
        self.latent_dim = latent_dim
        self.mapping = MappingNetwork(latent_dim)
        self.const_input = tf.Variable(tf.random.normal((1, 4, 4, channels[0])), trainable=True)
        self.blocks = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            self.blocks.append(GenBlock(in_ch, out_ch, upsample=True))
            in_ch = out_ch
        self.to_rgb = ModulatedConv2d(in_ch, 1, kernel_size=1, demodulate=False)

    def call(self, z):
        w = self.mapping(z)
        x = tf.tile(self.const_input, [tf.shape(z)[0], 1, 1, 1])
        for block in self.blocks:
            x = block(x, w)
        return self.to_rgb(x, w)


class Discriminator(tf.keras.Model):
    def __init__(self, in_ch=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 4, strides=2, padding="same")
        self.act1 = tf.keras.layers.LeakyReLU(0.2)
        self.conv2 = tf.keras.layers.Conv2D(64, 4, strides=2, padding="same")
        self.act2 = tf.keras.layers.LeakyReLU(0.2)
        self.conv3 = tf.keras.layers.Conv2D(128, 4, strides=2, padding="same")
        self.act3 = tf.keras.layers.LeakyReLU(0.2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.flatten(x)
        return self.fc(x)


def r1_penalty(discriminator, real_images):
    with tf.GradientTape() as tape:
        tape.watch(real_images)
        real_logits = discriminator(real_images)
        real_sum = tf.reduce_sum(real_logits)
    grads = tape.gradient(real_sum, real_images)
    return tf.reduce_mean(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))


def d_loss_fn(real_logits, fake_logits):
    real_loss = tf.reduce_mean(tf.nn.softplus(-real_logits))
    fake_loss = tf.reduce_mean(tf.nn.softplus(fake_logits))
    return real_loss + fake_loss


def g_loss_fn(fake_logits):
    return tf.reduce_mean(tf.nn.softplus(-fake_logits))


def make_train_step(generator, discriminator, g_opt, d_opt):
    # @tf.function traces this once into a graph instead of re-running the Python
    # loop body (and re-tracing every layer call) on every single batch. This is
    # the single biggest speed lever here, on top of the vectorized conv above.
    @tf.function
    def train_step(real_images, step):
        batch_size = tf.shape(real_images)[0]
        z = tf.random.normal((batch_size, LATENT_DIM))

        with tf.GradientTape() as d_tape:
            fake_images = generator(z)
            real_logits = discriminator(real_images)
            fake_logits = discriminator(fake_images)
            d_loss = d_loss_fn(real_logits, fake_logits)
            if tf.equal(step % D_REG, 0):
                d_loss = d_loss + (GAMMA / 2) * r1_penalty(discriminator, real_images)
        d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
        d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

        z = tf.random.normal((batch_size, LATENT_DIM))
        with tf.GradientTape() as g_tape:
            fake_images = generator(z)
            fake_logits = discriminator(fake_images)
            g_loss = g_loss_fn(fake_logits)
        g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
        g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))

        return g_loss, d_loss

    return train_step


def train(generator, discriminator, dataset, epochs=EPOCHS, lr=LR):
    g_opt = tf.keras.optimizers.Adam(lr, beta_1=0.0, beta_2=0.99)
    d_opt = tf.keras.optimizers.Adam(lr, beta_1=0.0, beta_2=0.99)
    train_step = make_train_step(generator, discriminator, g_opt, d_opt)

    z_ref = tf.random.normal((16, LATENT_DIM), seed=42)
    g_losses, d_losses = [], []
    step = 0
    snapshot_epochs = {1, 5, 10, 20, epochs}

    for epoch in range(1, epochs + 1):
        e_g, e_d = [], []
        for batch_idx, real_images in enumerate(dataset):
            step += 1
            g_loss, d_loss = train_step(real_images, tf.constant(step, dtype=tf.int64))
            e_g.append(float(g_loss)); e_d.append(float(d_loss))

            if batch_idx % 50 == 0:
                print(f"  epoch {epoch} step {batch_idx}  G={float(g_loss):.4f}  D={float(d_loss):.4f}")

        g_losses.append(np.mean(e_g)); d_losses.append(np.mean(e_d))
        print(f"Epoch {epoch}/{epochs}  G_loss={g_losses[-1]:.4f}  D_loss={d_losses[-1]:.4f}")

        if epoch in snapshot_epochs:
            save_grid(generator(z_ref), os.path.join(OUT_DIR, f"epoch_{epoch}.png"), f"Epoch {epoch}")

    return g_losses, d_losses


def save_grid(images, path, title, n_cols=4):
    images = np.clip((images.numpy() + 1) / 2, 0, 1)
    n = images.shape[0]
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    for i, ax in enumerate(np.array(axes).reshape(-1)):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i, ..., 0], cmap="gray")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def plot_losses(g_losses, d_losses, path=os.path.join(OUT_DIR, "loss_curves.png")):
    plt.figure(figsize=(7, 5))
    plt.plot(g_losses, label="Generator loss")
    plt.plot(d_losses, label="Discriminator loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("StyleGAN2 (Fashion-MNIST) training loss")
    plt.legend()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    x_train = load_fashion_mnist()
    dataset = make_dataset(x_train)

    generator = Generator()
    discriminator = Discriminator()

    g_losses, d_losses = train(generator, discriminator, dataset, epochs=EPOCHS)
    plot_losses(g_losses, d_losses)

    generator.save_weights(os.path.join(OUT_DIR, "generator.weights.h5"))
    discriminator.save_weights(os.path.join(OUT_DIR, "discriminator.weights.h5"))