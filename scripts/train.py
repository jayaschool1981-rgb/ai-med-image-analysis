import os, math
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

tf.random.set_seed(42)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
DATA_ROOT = os.environ.get("DATA_ROOT", "data/samples")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "models/v1")
EPOCHS_FROZEN = 5
EPOCHS_FINETUNE = 10

def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_ROOT, "train"),
        labels="inferred",
        label_mode="binary",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_ROOT, "val"),
        labels="inferred",
        label_mode="binary",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_ROOT, "test"),
        labels="inferred",
        label_mode="binary",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        shuffle=False,
    )
    AUTOTUNE = tf.data.AUTOTUNE
    return (
        train_ds.map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y)).prefetch(AUTOTUNE),
        val_ds.map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y)).prefetch(AUTOTUNE),
        test_ds.map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y)).prefetch(AUTOTUNE),
        train_ds.class_names
    )

def augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.15),
    ])

def build_model():
    base = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=IMAGE_SIZE + (3,)
    )
    base.trainable = False
    inputs = layers.Input(shape=IMAGE_SIZE + (3,))
    x = augmentation()(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc"),
                           tf.keras.metrics.BinaryAccuracy(name="acc"),
                           tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])
    return model, base

def compute_weights(ds):
    y_all = []
    for _, y in ds.unbatch():
        y_all.append(int(y.numpy()))
    y_all = np.array(y_all)
    classes = np.unique(y_all)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_all)
    return {int(c): float(w) for c, w in zip(classes, weights)}

def main():
    train_ds, val_ds, test_ds, class_names = build_datasets()
    model, base = build_model()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    class_weights = compute_weights(train_ds)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, "best.h5"),
                                           monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(os.path.join(OUTPUT_DIR, "training_log.csv")),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FROZEN,
              class_weight=class_weights, callbacks=callbacks)

    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc"),
                           tf.keras.metrics.BinaryAccuracy(name="acc"),
                           tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE,
              class_weight=class_weights, callbacks=callbacks)

    # Evaluate
    eval_res = model.evaluate(test_ds, return_dict=True)
    print("Test metrics:", eval_res)

    # Save final
    model.save(os.path.join(OUTPUT_DIR, "model.h5"))
    model.save(os.path.join(OUTPUT_DIR, "saved_model"))

if __name__ == "__main__":
    main()
