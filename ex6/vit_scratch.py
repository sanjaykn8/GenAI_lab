import numpy as np
import tensorflow as tf

IMG_SIZE = 32
PATCH_SIZE = 4
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
CHANNELS = 3
D_MODEL = 192
NUM_HEADS = 6
HEAD_DIM = D_MODEL // NUM_HEADS
MLP_DIM = D_MODEL * 4
NUM_LAYERS = 6
NUM_CLASSES = 10
DROPOUT_RATE = 0.1


class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size=PATCH_SIZE, d_model=D_MODEL):
        super().__init__()
        self.patch_size = patch_size
        self.proj = tf.keras.layers.Conv2D(d_model, kernel_size=patch_size, strides=patch_size)
        self.class_token = tf.Variable(tf.random.normal((1, 1, d_model)) * 0.02, trainable=True)
        self.pos_embed = tf.Variable(tf.random.normal((1, NUM_PATCHES + 1, d_model)) * 0.02, trainable=True)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        patches = self.proj(x)
        patches = tf.reshape(patches, (batch_size, -1, patches.shape[-1]))
        cls = tf.tile(self.class_token, [batch_size, 1, 1])
        tokens = tf.concat([cls, patches], axis=1)
        return tokens + self.pos_embed


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model=D_MODEL, num_heads=NUM_HEADS, head_dim=HEAD_DIM):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.wq = tf.keras.layers.Dense(inner_dim)
        self.wk = tf.keras.layers.Dense(inner_dim)
        self.wv = tf.keras.layers.Dense(inner_dim)
        self.wo = tf.keras.layers.Dense(d_model)
        self.last_attn_weights = None

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, [0, 2, 1, 3])

    def call(self, x):
        batch_size = tf.shape(x)[0]
        q = self.split_heads(self.wq(x), batch_size)
        k = self.split_heads(self.wk(x), batch_size)
        v = self.split_heads(self.wv(x), batch_size)

        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.head_dim, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        self.last_attn_weights = weights

        out = tf.matmul(weights, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, (batch_size, -1, self.num_heads * self.head_dim))
        return self.wo(out)


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model=D_MODEL, mlp_dim=MLP_DIM, dropout=DROPOUT_RATE):
        super().__init__()
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.mha = MultiHeadSelfAttention(d_model)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.fc1 = tf.keras.layers.Dense(mlp_dim, activation="gelu")
        self.fc2 = tf.keras.layers.Dense(d_model)
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        h = self.mha(self.ln1(x))
        x = x + self.drop1(h, training=training)
        h = self.fc2(self.fc1(self.ln2(x)))
        x = x + self.drop2(h, training=training)
        return x


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers=NUM_LAYERS):
        super().__init__()
        self.blocks = [TransformerEncoderLayer() for _ in range(num_layers)]

    def call(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)
        return x

    def get_attention_maps(self):
        return [b.mha.last_attn_weights for b in self.blocks]


class ViT(tf.keras.Model):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.patch_embed = PatchEmbedding()
        self.encoder = TransformerEncoder()
        self.final_ln = tf.keras.layers.LayerNormalization()
        self.head = tf.keras.layers.Dense(num_classes)

    def call(self, x, training=False):
        tokens = self.patch_embed(x)
        tokens = self.encoder(tokens, training=training)
        cls_out = self.final_ln(tokens)[:, 0]
        return self.head(cls_out)

    def get_attention_maps(self):
        return self.encoder.get_attention_maps()


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = (x_train.astype("float32") / 127.5) - 1.0
    x_test = (x_test.astype("float32") / 127.5) - 1.0
    y_train = y_train.squeeze(-1)
    y_test = y_test.squeeze(-1)
    return (x_train, y_train), (x_test, y_test)


def train_vit(model, x_train, y_train, x_val, y_val, epochs=20, batch_size=128, lr=3e-4):
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=epochs, batch_size=batch_size)
    return history


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    model = ViT()
    _ = model(x_train[:2])
    model.summary()
    train_vit(model, x_train, y_train, x_test, y_test, epochs=1, batch_size=128)
