import numpy as np
import tensorflow as tf

IMG_SIZE = 28
PATCH_SIZE = 4
GRID_SIZE = IMG_SIZE // PATCH_SIZE
NUM_PATCHES = GRID_SIZE * GRID_SIZE
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 1
D_MODEL = 128
NUM_HEADS = 4
HEAD_DIM = D_MODEL // NUM_HEADS
MLP_DIM = D_MODEL * 4
NUM_LAYERS = 4
NUM_CLASSES = 10
DROPOUT_RATE = 0.1
SEQ_LEN = NUM_PATCHES + 1  # +1 for the prepended class-condition token


def image_to_patches(images, patch_size=PATCH_SIZE):
    b = tf.shape(images)[0]
    patches = tf.image.extract_patches(
        images, sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1], rates=[1, 1, 1, 1], padding="VALID")
    return tf.reshape(patches, (b, -1, patch_size * patch_size))


def patches_to_image(patches, grid_size=GRID_SIZE, patch_size=PATCH_SIZE):
    b = tf.shape(patches)[0]
    patches = tf.reshape(patches, (b, grid_size, grid_size, patch_size, patch_size))
    patches = tf.transpose(patches, [0, 1, 3, 2, 4])
    return tf.reshape(patches, (b, grid_size * patch_size, grid_size * patch_size, 1))


def causal_mask(seq_len):
    return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)


class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model=D_MODEL, num_heads=NUM_HEADS, head_dim=HEAD_DIM):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim
        self.wq = tf.keras.layers.Dense(inner)
        self.wk = tf.keras.layers.Dense(inner)
        self.wv = tf.keras.layers.Dense(inner)
        self.wo = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, b):
        x = tf.reshape(x, (b, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, [0, 2, 1, 3])

    def call(self, x, mask):
        b = tf.shape(x)[0]
        q = self.split_heads(self.wq(x), b)
        k = self.split_heads(self.wk(x), b)
        v = self.split_heads(self.wv(x), b)

        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.head_dim, tf.float32))
        scores = scores + (mask * -1e9)
        weights = tf.nn.softmax(scores, axis=-1)

        out = tf.matmul(weights, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, (b, -1, self.num_heads * self.head_dim))
        return self.wo(out)


class CausalTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model=D_MODEL, mlp_dim=MLP_DIM, dropout=DROPOUT_RATE):
        super().__init__()
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.attn = CausalSelfAttention(d_model)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.fc1 = tf.keras.layers.Dense(mlp_dim, activation="gelu")
        self.fc2 = tf.keras.layers.Dense(d_model)
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask, training=False):
        h = self.attn(self.ln1(x), mask)
        x = x + self.drop1(h, training=training)
        h = self.fc2(self.fc1(self.ln2(x)))
        x = x + self.drop2(h, training=training)
        return x


class CausalTransformer(tf.keras.layers.Layer):
    def __init__(self, num_layers=NUM_LAYERS):
        super().__init__()
        self.blocks = [CausalTransformerBlock() for _ in range(num_layers)]

    def call(self, x, mask, training=False):
        for block in self.blocks:
            x = block(x, mask, training=training)
        return x


class ConditionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_classes=NUM_CLASSES, d_model=D_MODEL):
        super().__init__()
        self.cond_embed = tf.keras.layers.Embedding(num_classes, d_model)

    def call(self, labels):
        return self.cond_embed(labels)[:, None, :]


class AutoregressiveDecoder(tf.keras.Model):
    def __init__(self, num_layers=NUM_LAYERS):
        super().__init__()
        self.patch_proj = tf.keras.layers.Dense(D_MODEL)
        self.pos_embed = tf.Variable(tf.random.normal((1, NUM_PATCHES, D_MODEL)) * 0.02, trainable=True)
        self.cond_embed = ConditionalEmbedding()
        self.transformer = CausalTransformer(num_layers)
        self.final_ln = tf.keras.layers.LayerNormalization()
        self.out_proj = tf.keras.layers.Dense(PATCH_DIM, activation="tanh")

    def call(self, patches, labels, training=False):
        b = tf.shape(patches)[0]
        patch_tokens = self.patch_proj(patches) + self.pos_embed  # (B, N, D)
        cond_token = self.cond_embed(labels)  # (B, 1, D)

        seq_in = tf.concat([cond_token, patch_tokens[:, :-1, :]], axis=1)  # (B, N, D)
        mask = causal_mask(NUM_PATCHES)
        h = self.transformer(seq_in, mask, training=training)
        h = self.final_ln(h)
        return self.out_proj(h)  # predicted patches[0..N-1], (B, N, PATCH_DIM)

    def generate(self, labels, num_patches=NUM_PATCHES):
        b = tf.shape(labels)[0]
        cond_token = self.cond_embed(labels)
        generated_patches = []
        seq = cond_token  # start with just the condition token, grows by one each step

        snapshots = {}
        for step in range(num_patches):
            cur_len = tf.shape(seq)[1]
            mask = causal_mask(cur_len)
            h = self.transformer(seq, mask, training=False)
            h = self.final_ln(h)
            next_patch = self.out_proj(h)[:, -1:, :]  # (B, 1, PATCH_DIM)
            generated_patches.append(next_patch[:, 0, :])

            next_token = self.patch_proj(next_patch) + self.pos_embed[:, step:step + 1, :]
            seq = tf.concat([seq, next_token], axis=1)

            if (step + 1) in {1, 10, 25, num_patches}:
                partial = tf.stack(generated_patches, axis=1)
                pad_len = num_patches - partial.shape[1]
                if pad_len > 0:
                    partial = tf.concat([partial, tf.zeros((b, pad_len, PATCH_DIM))], axis=1)
                snapshots[step + 1] = patches_to_image(partial)

        full = tf.stack(generated_patches, axis=1)
        return patches_to_image(full), snapshots


def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = (x_train.astype("float32") / 127.5) - 1.0
    x_test = (x_test.astype("float32") / 127.5) - 1.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)


def train_generator(model, x_train, y_train, epochs=20, batch_size=128, lr=3e-4):
    optimizer = tf.keras.optimizers.Adam(lr)
    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        for images, labels in ds:
            patches = image_to_patches(images)
            with tf.GradientTape() as tape:
                pred_patches = model(patches, labels, training=True)
                loss = tf.reduce_mean(tf.square(pred_patches - patches))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_losses.append(float(loss))
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")
    return losses


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    model = AutoregressiveDecoder()
    losses = train_generator(model, x_train, y_train, epochs=1, batch_size=64)

    labels = tf.constant([1, 9])  # Trouser, Ankle boot
    images, snapshots = model.generate(labels)
    print("generated:", images.shape)
    print("snapshot steps:", list(snapshots.keys()))
