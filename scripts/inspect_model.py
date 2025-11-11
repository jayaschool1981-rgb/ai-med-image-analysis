import tensorflow as tf

m = tf.keras.models.load_model("models/v1/model.h5", compile=False)
print("\n== Top-level layers ==")
for i, l in enumerate(m.layers):
    print(i, l.name, type(l).__name__)

try:
    base = m.get_layer("efficientnetb0")
    print("\n== efficientnetb0 conv tail ==")
    for l in reversed(base.layers):
        if isinstance(l, tf.keras.layers.Conv2D):
            print("Last conv layer:", l.name)
            break
except Exception as e:
    print("\nNo efficientnetb0 sublayer available:", e)
