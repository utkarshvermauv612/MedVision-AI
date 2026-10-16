
import numpy as np
import tensorflow as tf

MODEL_PATH = r"A:\Project\Medical_Image_Analysis\models\best_model.keras"

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("\n========== MODEL HEALTH CHECK ==========")

# 1. Check weights for invalid values
for layer in model.layers:

    weights = layer.get_weights()

    for index, weight in enumerate(weights):

        if weight.size == 0:
            continue

        print(
            f"{layer.name} | weight {index} | "
            f"min={np.min(weight):.6g} | "
            f"max={np.max(weight):.6g} | "
            f"mean={np.mean(weight):.6g} | "
            f"std={np.std(weight):.6g} | "
            f"finite={np.all(np.isfinite(weight))}"
        )

# 2. Extract the value before the final sigmoid
logit_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.layers[-1].input
)

# 3. Create a test batch with the correct input shape
# This checks the model numerically, not its medical accuracy.
dummy_images = tf.zeros((4, 256, 256, 1), dtype=tf.float32)

# 4. Inspect final-layer logits
logits = logit_model(dummy_images, training=False).numpy()

probabilities = model(dummy_images, training=False).numpy()

print("\n========== RAW OUTPUT ==========")
print("Logits:", logits.reshape(-1))
print("Probabilities:", probabilities.reshape(-1))

print("\nLogits finite:", np.all(np.isfinite(logits)))
print("Probabilities finite:", np.all(np.isfinite(probabilities)))

# 5. Check whether the output is saturated
print(
    "All probabilities approximately 1:",
    np.all(probabilities > 0.9999)
)

print("\nDiagnostic completed.")