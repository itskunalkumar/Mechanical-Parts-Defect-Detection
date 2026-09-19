import sys

import numpy as np
import tensorflow as tf
from PIL import Image

from src.exception import CustomException
from src.logger import logging

LAST_CONV_LAYER_CANDIDATES = ["Conv_1", "out_relu", "block_16_project"]


class GradCAM:
    """
    Generates a Grad-CAM heatmap showing which region of the image the model
    focused on to make its prediction. Works on top of the MobileNetV2 backbone
    used in ModelTrainer.
    """

    def __init__(self, model):
        self.model = model
        self.last_conv_layer_name = self._find_last_conv_layer()

    def _find_last_conv_layer(self):
        # MobileNetV2 is nested as the first layer of the Sequential model
        base_model = self.model.layers[0]
        for name in LAST_CONV_LAYER_CANDIDATES:
            try:
                base_model.get_layer(name)
                return name
            except ValueError:
                continue
        # fallback: last conv-like layer found by scanning
        for layer in reversed(base_model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find a convolutional layer for Grad-CAM")

    def compute_heatmap(self, img_array: np.ndarray) -> np.ndarray:
        try:
            base_model = self.model.layers[0]
            grad_model = tf.keras.models.Model(
                inputs=base_model.input,
                outputs=[base_model.get_layer(self.last_conv_layer_name).output, base_model.output],
            )

            with tf.GradientTape() as tape:
                conv_outputs, base_output = grad_model(img_array)
                # pass base_output through the rest of the model (head layers)
                x = base_output
                for layer in self.model.layers[1:]:
                    x = layer(x)
                prediction = x[:, 0]

            grads = tape.gradient(prediction, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
            return heatmap.numpy()

        except Exception as e:
            raise CustomException(e, sys)

    def overlay_heatmap(self, original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
        try:
            import matplotlib.cm as cm

            heatmap_resized = np.array(
                Image.fromarray(np.uint8(255 * heatmap)).resize(original_image.size)
            )
            jet = cm.get_cmap("jet")
            jet_colors = jet(np.arange(256))[:, :3]
            jet_heatmap = jet_colors[heatmap_resized]
            jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255)).convert("RGB")

            overlaid = Image.blend(original_image.convert("RGB"), jet_heatmap, alpha)
            return overlaid

        except Exception as e:
            raise CustomException(e, sys)
