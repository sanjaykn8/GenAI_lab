import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import TFViTModel

# requires: pip install "transformers==4.46.3" tf-keras
# (newer transformers dropped TF model classes; tf-keras is the Keras-2 shim
# transformers needs to run under Keras 3 environments)

IMG_SIZE = 224
NUM_CLASSES = 37
BATCH_SIZE = 16
CHECKPOINT = "google/vit-base-patch16-224-in21k"
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406])
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225])


def preprocess(example, training=True):
    img = tf.cast(example["image"], tf.float32) / 255.0
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    if training:
        img = tf.image.random_flip_left_right(img)
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = tf.transpose(img, [2, 0, 1])  # HF ViT expects channels-first pixel_values
    return img, example["label"]


def load_oxford_pets(batch_size=BATCH_SIZE):
    train_ds = tfds.load("oxford_iiit_pet", split="train")
    test_ds = tfds.load("oxford_iiit_pet", split="test")
    train_ds = (train_ds
            .map(lambda e: preprocess(e, True), num_parallel_calls=2)
            .shuffle(1000)
            .batch(batch_size)
            .prefetch(1))
    test_ds = (test_ds
           .map(lambda e: preprocess(e, False), num_parallel_calls=2)
           .batch(batch_size)
           .prefetch(1))
    return train_ds, test_ds


class PretrainedViTClassifier(tf.keras.Model):
    def __init__(self, num_classes=NUM_CLASSES, checkpoint=CHECKPOINT):
        super().__init__()
        self.backbone = TFViTModel.from_pretrained(checkpoint)
        self.head = tf.keras.layers.Dense(num_classes)

    def call(self, pixel_values, training=False):
        out = self.backbone(pixel_values=pixel_values, training=training)
        cls_token = out.last_hidden_state[:, 0]  # z_L^0
        return self.head(cls_token)


def set_backbone_trainable(model, trainable, unfreeze_last_k=0):
    encoder_layers = model.backbone.vit.encoder.layer
    for layer in encoder_layers:
        layer.trainable = False
    model.backbone.vit.embeddings.trainable = False
    # the pooler is never called (we use last_hidden_state[:, 0] instead), so its
    # weights would otherwise sit in backbone.trainable_variables with no gradient
    # ever computed for them, triggering a spurious "no gradient" warning every step
    model.backbone.vit.pooler.trainable = False
    if trainable and unfreeze_last_k > 0:
        for layer in encoder_layers[-unfreeze_last_k:]:
            layer.trainable = True


def linear_probe(model, train_ds, val_ds, epochs=5, lr=1e-3, log_every=20):
    set_backbone_trainable(model, trainable=False)
    optimizer = tf.keras.optimizers.Adam(lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # @tf.function traces the step into a graph once instead of re-running the full
    # ViT-Base forward pass in eager mode on every single batch — this is the main
    # speed lever here, on top of removing the pooler-warning overhead above.
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.head.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.head.trainable_variables))
        acc_metric.update_state(y, logits)
        return loss

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    for epoch in range(epochs):
        acc_metric.reset_state()
        epoch_losses = []
        for step, (x, y) in enumerate(train_ds):
            loss = train_step(x, y)
            epoch_losses.append(float(loss))
            if step % log_every == 0:
                print(f"  [linear probe] epoch {epoch + 1} step {step}  loss={float(loss):.4f}")
        train_loss, train_acc = np.mean(epoch_losses), float(acc_metric.result())

        acc_metric.reset_state()
        val_losses = []
        for x, y in val_ds:
            logits = model(x, training=False)
            val_losses.append(float(loss_fn(y, logits)))
            acc_metric.update_state(y, logits)
        val_loss, val_acc = np.mean(val_losses), float(acc_metric.result())

        history["loss"].append(train_loss); history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss); history["val_accuracy"].append(val_acc)
        print(f"[linear probe] epoch {epoch + 1}/{epochs}  loss={train_loss:.4f} acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    return history


def selective_finetune(
    model,
    train_ds,
    val_ds,
    epochs=2,
    lr_backbone=1e-5,
    lr_head=1e-4,
    unfreeze_last_k=1,
    log_every=10
):
    set_backbone_trainable(
        model,
        trainable=True,
        unfreeze_last_k=unfreeze_last_k
    )

    backbone_vars = list(model.backbone.trainable_variables)
    head_vars = list(model.head.trainable_variables)
    all_vars = backbone_vars + head_vars

    backbone_opt = tf.keras.optimizers.Adam(learning_rate=lr_backbone)
    head_opt = tf.keras.optimizers.Adam(learning_rate=lr_head)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)

        # One backward pass instead of two persistent-tape passes
        all_grads = tape.gradient(loss, all_vars)

        n_backbone = len(backbone_vars)

        backbone_grads = all_grads[:n_backbone]
        head_grads = all_grads[n_backbone:]

        backbone_opt.apply_gradients(
            zip(backbone_grads, backbone_vars)
        )

        head_opt.apply_gradients(
            zip(head_grads, head_vars)
        )

        acc_metric.update_state(y, logits)

        return loss

    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(epochs):

        acc_metric.reset_state()
        epoch_losses = []

        print(f"\n[fine-tune] Starting epoch {epoch + 1}/{epochs}")

        for step, (x, y) in enumerate(train_ds):

            loss = train_step(x, y)
            epoch_losses.append(float(loss))

            if step % log_every == 0:
                print(
                    f"  [fine-tune] "
                    f"epoch {epoch + 1} "
                    f"step {step} "
                    f"loss={float(loss):.4f}"
                )

        train_loss = np.mean(epoch_losses)
        train_acc = float(acc_metric.result())

        # Validation
        acc_metric.reset_state()
        val_losses = []

        for x, y in val_ds:
            logits = model(x, training=False)

            val_loss_batch = loss_fn(y, logits)
            val_losses.append(float(val_loss_batch))

            acc_metric.update_state(y, logits)

        val_loss = np.mean(val_losses)
        val_acc = float(acc_metric.result())

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"[fine-tune] epoch {epoch + 1}/{epochs} "
            f"loss={train_loss:.4f} "
            f"acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

    return history


def get_predictions(model, dataset):
    y_true, y_pred = [], []
    for x, y in dataset:
        logits = model(x, training=False)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(tf.argmax(logits, axis=-1).numpy().tolist())
    return np.array(y_true), np.array(y_pred)


if __name__ == "__main__":
    train_ds, val_ds = load_oxford_pets()
    model = PretrainedViTClassifier()

    print("=== Stage 1: linear probing ===")
    linear_probe(model, train_ds, val_ds, epochs=2)

    print("=== Stage 2: selective fine-tuning ===")
    selective_finetune(model, train_ds, val_ds, epochs=2)
