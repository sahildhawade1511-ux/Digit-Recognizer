import gradio as gr
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('my_mnist_model.keras')

def preprocess_image(image):

    if image.shape[-1] == 4:
        image = image[..., :3]


    gray = tf.image.rgb_to_grayscale(image)
    gray = 255 - gray

    gray_np = tf.squeeze(gray).numpy().astype(np.uint8)

    coords = np.column_stack(np.where(gray_np > 0))
    if coords.size == 0:
        return np.zeros((28, 28, 1), dtype=np.float32) 

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = gray_np[y_min:y_max+1, x_min:x_max+1]
    # Resize cropped digit
    resized = tf.image.resize(cropped[..., np.newaxis], (20, 20)).numpy()
    padded = np.pad(resized, ((4, 4), (4, 4), (0, 0)), mode='constant', constant_values=0)
    padded = padded / 255.0

    return padded


def predict_digit(drawing_dict):
    if drawing_dict is None:
        return "Please draw a digit first."

    processed_image = preprocess_image(drawing_dict['composite'])
    processed_image = processed_image.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(processed_image)
    confidences = {str(i): float(prediction[0][i]) for i in range(10)}
    return confidences

interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(type="numpy"),
    outputs=gr.Label(num_top_classes=3),
    title="Digit Recognizer",
    description="Draw a digit (0-9)."
)

print("Launching the final Gradio interface...")
interface.launch()
