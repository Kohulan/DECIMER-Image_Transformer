import tensorflow as tf
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

import re
import sys
import time
import pickle
import I2S_Model_Transformer
from datetime import datetime

tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="node-blur")
print("Running on TPU ", tpu.master())

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)

f = open("Training_Report.txt", "w")
print("Number of devices: {}".format(strategy.num_replicas_in_sync), flush=True)
sys.stdout = f
print("REPLICAS: ", strategy.num_replicas_in_sync, flush=True)
print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "Network Started", flush=True)

# Load the Data
total_data = ...  # Number of total train datasize

tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
max_length = pickle.load(open("max_length.pkl", "rb"))

# Parameters to train the network

# Set Training Epochs
EPOCHS = 60
BATCH_SIZE = 128 * strategy.num_replicas_in_sync
BUFFER_SIZE = 10000
target_vocab_size = len(tokenizer.word_index) + 1
num_steps = total_data // BATCH_SIZE
num_layer = 4
d_model = 512
dff = 2048
num_heads = 8
row_size = 10
col_size = 10
dropout_rate = 0.1

print("Total train steps: ", num_steps)

AUTO = tf.data.experimental.AUTOTUNE


def read_tfrecord(example):
    feature = {
        "image_raw": tf.io.FixedLenFeature([], tf.string),
        "caption": tf.io.FixedLenFeature([], tf.string),
    }

    # decode the TFRecord
    example = tf.io.parse_single_example(example, feature)

    img = tf.io.decode_raw(example["image_raw"], tf.float32)
    img_tensor = tf.reshape(img, [100, 1536])
    caption = tf.io.decode_raw(example["caption"], tf.int32)
    # caption = tf.reshape(capt, [42,])

    return img_tensor, caption


numbers = re.compile(r"(\d+)")


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def get_training_dataset(batch_size=BATCH_SIZE, buffered_size=BUFFER_SIZE):
    options = tf.data.Options()
    filenames = sorted(
        tf.io.gfile.glob("path/to/TFRecord/*.tfrecord"), key=numericalSort
    )
    # print(len(filenames))

    dataset_img = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    train_dataset = (
        dataset_img.with_options(options)
        .map(read_tfrecord, num_parallel_calls=AUTO)
        .repeat()
        .shuffle(buffered_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=AUTO)
    )
    return train_dataset


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_masks_decoder(tar):
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return combined_mask


with strategy.scope():
    # Network Parameters
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    train_loss = tf.keras.metrics.Mean(name="train_loss", dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name="train_accuracy", dtype=tf.float32
    )

    # Initialize Transformer
    transformer = I2S_Model_Transformer.Transformer(
        num_layer,
        d_model,
        num_heads,
        dff,
        row_size,
        col_size,
        target_vocab_size,
        max_pos_encoding=target_vocab_size,
        rate=dropout_rate,
    )

checkpoint_path = "path/to/checkpoint"
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
    start_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])

per_replica_batch_size = BATCH_SIZE // strategy.num_replicas_in_sync

print("Batch Size", BATCH_SIZE)
print("Per replica", per_replica_batch_size)
train_dataset = strategy.experimental_distribute_dataset(get_training_dataset())

# the loss_plot array will be reset many times
loss_plot = []
accuracy_plot = []


@tf.function
def train_step(iterator):
    def step_fn(inputs):
        img_tensor, target = inputs
        loss = 0

        tar_inp = target[:, :-1]
        tar_real = target[:, 1:]

        dec_mask = create_masks_decoder(tar_inp)
        print(dec_mask)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(img_tensor, tar_inp, True, dec_mask)
            loss = loss_function(tar_real, predictions)

        total_loss = loss / int(target.shape[1])

        gradients = tape.gradient(loss, transformer.trainable_variables)

        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss.update_state(loss * strategy.num_replicas_in_sync)
        train_accuracy.update_state(tar_real, predictions)

        return loss, total_loss

    per_replica_losses, l_loss = strategy.run(step_fn, args=(iterator,))
    return strategy.reduce(
        tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None
    ), strategy.reduce(tf.distribute.ReduceOp.MEAN, l_loss, axis=None)


print(
    datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "Actual Training Started", flush=True
)

train_iterator = iter(train_dataset)

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0
    batch = 0
    train_loss.reset_states()
    train_accuracy.reset_states()
    print(str(learning_rate))

    for x in train_dataset:
        img_tensor, target = x
        batch_loss, t_loss = train_step(x)
        total_loss += t_loss
        batch += 1

        if batch % 100 == 0:
            print(
                "Epoch {} Batch {} Loss {:.4f} Updated_Loss {:.4f} Accuracy {:.4f}".format(
                    epoch + 1,
                    batch,
                    batch_loss.numpy() / BATCH_SIZE,
                    train_loss.result(),
                    train_accuracy.result(),
                ),
                flush=True,
            )
        # storing the epoch end loss value to plot later

        if batch == num_steps:
            loss_plot.append(total_loss / num_steps)
            ckpt_manager.save()

            print(
                "Epoch {} Loss {:.6f} Updated_Loss {:.4f} Accuracy {:.4f}".format(
                    epoch + 1,
                    total_loss / num_steps,
                    train_loss.result(),
                    train_accuracy.result(),
                ),
                flush=True,
            )
            print(
                datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                "Time taken for 1 epoch {} sec\n".format(time.time() - start),
                flush=True,
            )
            # transformer.save_weights('Epoch_'+str(epoch+1)+'_weights.h5')

            break

plt.plot(loss_plot, "-o", label="Loss value")
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
# plt.show()
plt.gcf().set_size_inches(20, 20)
plt.savefig("Lossplot_.jpg")
plt.close()

print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "Network Completed", flush=True)
f.close()
