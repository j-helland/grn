import argparse

import glb


@glb.gpu_load_balance()
def train_job(epochs: int, batch_size: int) -> 'tf.keras.Model':
    import tensorflow as tf
            
    # Configure tensorflow to progressively allocate GPU memory rather than fully 
    # allocating all available memory.
    physical_devices = tf.config.list_physical_devices('GPU') 
    for gpu_instance in physical_devices: 
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Build the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    model.build(input_shape=(None, 28, 28))
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    model.fit(
        x_train, y_train,
        epochs=epochs, 
        batch_size=batch_size,
        validation_data=(x_test, y_test))  # Bad form to use test as val, but this is just a load-balancing test.

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    trained_model = train_job(
        epochs=args.epochs, 
        batch_size=args.batch_size)
