"""
CNN architecture for Traffic Sign Recognition.

Dataset target: GTSRB or similar traffic sign datasets.
Input shape: 32x32x3.
Default number of classes for GTSRB: 43.
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam


def build_basic_model(input_shape=(32, 32, 3), num_classes=43):
    """
    Build a basic CNN model for traffic sign classification.

    This version is simple, easy to explain, and suitable as a baseline.
    """
    model = Sequential(
        [
            Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                input_shape=input_shape,
                name="conv1_32",
            ),
            MaxPooling2D(pool_size=(2, 2), name="pool1"),
            Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2_64"),
            MaxPooling2D(pool_size=(2, 2), name="pool2"),
            Flatten(name="flatten"),
            Dense(128, activation="relu", name="fc1_128"),
            Dropout(0.5, name="dropout_50"),
            Dense(num_classes, activation="softmax", name="predictions"),
        ],
        name="Basic_TrafficSign_CNN",
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_improved_model(input_shape=(32, 32, 3), num_classes=43):
    """
    Build an improved CNN model for traffic sign classification.

    This version uses deeper convolution blocks and dropout to improve
    feature extraction while reducing overfitting.
    """
    model = Sequential(
        [
            Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                input_shape=input_shape,
                name="block1_conv1_32",
            ),
            Conv2D(32, (3, 3), activation="relu", padding="same", name="block1_conv2_32"),
            MaxPooling2D(pool_size=(2, 2), name="block1_pool"),
            Dropout(0.25, name="block1_dropout_25"),
            Conv2D(64, (3, 3), activation="relu", padding="same", name="block2_conv1_64"),
            Conv2D(64, (3, 3), activation="relu", padding="same", name="block2_conv2_64"),
            MaxPooling2D(pool_size=(2, 2), name="block2_pool"),
            Dropout(0.25, name="block2_dropout_25"),
            Conv2D(128, (3, 3), activation="relu", padding="same", name="block3_conv1_128"),
            MaxPooling2D(pool_size=(2, 2), name="block3_pool"),
            Dropout(0.25, name="block3_dropout_25"),
            Flatten(name="flatten"),
            Dense(256, activation="relu", name="fc1_256"),
            Dropout(0.5, name="fc_dropout_50"),
            Dense(num_classes, activation="softmax", name="predictions"),
        ],
        name="Improved_TrafficSign_CNN",
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(input_shape=(32, 32, 3), num_classes=43, model_type="improved"):
    """
    Build a CNN model.

    Args:
        input_shape: Input image shape. Default is 32x32x3.
        num_classes: Number of output classes. GTSRB has 43 classes.
        model_type: "basic" or "improved".
    """
    if model_type == "basic":
        return build_basic_model(input_shape=input_shape, num_classes=num_classes)
    if model_type == "improved":
        return build_improved_model(input_shape=input_shape, num_classes=num_classes)

    raise ValueError("model_type must be either 'basic' or 'improved'")


if __name__ == "__main__":
    model = build_model()
    model.summary()
