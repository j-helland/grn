import os
import argparse

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

import glb


class Model(nn.Module):
    training: bool

    @nn.compact
    def __call__(self, x):
        x = x.reshape((len(x), -1)) # flatten
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.2)(x, deterministic=not self.training)
        x = nn.Dense(features=10)(x)
        return x


@glb.job()
def train_job(epochs: int, batch_size: int) -> dict:
    # Force JAX to not preallocate 90% of the GPU.
    # Instead use a dynamic growth policy.
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"

    # Normally we would prefer to use tensorflow_datasets, but 
    # we already have tensorflow installed for this example.
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = jnp.float32(x_train) / 255.0
    x_test = jnp.float32(x_test) / 255.0

    def cross_entropy_loss(log_scores, labels):
        one_hot = jax.nn.one_hot(labels, num_classes=10)
        return -jnp.mean(jnp.sum(one_hot * log_scores, axis=-1))

    def cross_entropy_from_logits_loss(logits, labels):
        return cross_entropy_loss(nn.log_softmax(logits), labels)

    def compute_metrics(logits, labels) -> dict:
        loss = cross_entropy_from_logits_loss(logits, labels)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        return dict(
            loss=loss, 
            accuracy=accuracy)

    def create_train_state(rng, dropout_rng, learning_rate=1e-3):
        model = Model(training=True)
        params = model.init(
            {'params': rng, 'dropout': dropout_rng}, 
            jnp.ones([1, 28, 28, 1]))['params']
        optimizer = optax.adam(learning_rate)
        return train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer)

    @jax.jit
    def train_step(state, batch, dropout_rng):
        def loss_func(params):
            logits = Model(training=True).apply(
                {'params': params}, batch[0], 
                rngs={'dropout': dropout_rng})
            loss = cross_entropy_from_logits_loss(logits, labels=batch[1])
            return loss, logits
        
        grad_func = jax.value_and_grad(loss_func, has_aux=True)
        (_, logits), grads = grad_func(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = compute_metrics(logits, labels=batch[1])
        return state, metrics
    
    @jax.jit
    def eval_step(params, batch):
        logits = Model(training=False).apply({'params': params}, batch[0])
        return compute_metrics(logits, labels=batch[1])

    def train_epoch(state, batch_size, rng, dropout_rng):
        train_ds_size = len(x_train)
        steps_per_epoch = train_ds_size // batch_size

        perms = jax.random.permutation(rng, train_ds_size)
        perms = perms[:steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))
        batch_metrics = []
        for perm in perms:
            batch = (x_train[perm], y_train[perm])
            state, metrics = train_step(state, batch, dropout_rng)
            batch_metrics.append(metrics)

        return state
    
    def eval_model(params, x, y):
        metrics = eval_step(params, (x, y))
        metrics = jax.device_get(metrics)
        summary = jax.tree_map(lambda x: x.item(), metrics)
        return summary['loss'], summary['accuracy']

    rng = jax.random.PRNGKey(0)
    rng, init_rng, init_dropout_rng = jax.random.split(rng, 3)

    state = create_train_state(init_rng, init_dropout_rng, learning_rate=1e-3)
    del init_rng

    epochs = 1
    batch_size = 128
    for epoch in range(epochs):
        # Create separate PRNG key for shuffling data.
        rng, input_rng, dropout_rng = jax.random.split(rng, 3)
        state = train_epoch(state, batch_size, input_rng, dropout_rng)

        train_loss, train_acc = eval_model(state.params, x_train, y_train)
        test_loss, test_acc = eval_model(state.params, x_test, y_test)
        print(
            f'Epoch {epoch + 1} / {epochs} | '
            f'train loss {train_loss}, acc {train_acc} | '
            f'test loss {test_loss}, acc {test_acc}')


@train_job.profile
def _():
    print('I am a special version of the job used for profiling.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    trained_params = train_job(
        epochs=args.epochs, 
        batch_size=args.batch_size)
