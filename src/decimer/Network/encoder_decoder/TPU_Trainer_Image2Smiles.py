import tensorflow as tf
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

import re
import sys
import time
import pickle
import I2S_Model
from datetime import datetime

np.set_printoptions(threshold=sys.maxsize)

tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="node-3")
print("Running on TPU ", tpu.master())

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)

f = open("Training_900k1v3-32_128_single_TPU.txt", "w")
print("Number of devices: {}".format(strategy.num_replicas_in_sync), flush=True)
sys.stdout = f
print("REPLICAS: ", strategy.num_replicas_in_sync, flush=True)
print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "Network Started", flush=True)

# Load the Data
PATH = "gs://tpu-test-koh/Train_Images/"
total_data = 999424

tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
max_length = pickle.load(open("max_length.pkl", "rb"))

# Parameters to train the network

# Set Training Epochs
EPOCHS = 40
BATCH_SIZE = 128 * strategy.num_replicas_in_sync
# BATCH_SIZE = 8
BUFFER_SIZE = 1000
embedding_dim = 600
units = 1024
vocab_size = len(tokenizer.word_index) + 1
num_steps = total_data // BATCH_SIZE

# shape of the vector extracted from InceptionV3 is (64, 2048) these two variables represent that
features_shape = 2048
attention_features_shape = 64

AUTO = tf.data.experimental.AUTOTUNE


def read_tfrecord(example):
    feature = {
        # 'image_id': tf.io.FixedLenFeature([], tf.string),
        "image_raw": tf.io.FixedLenFeature([], tf.string),
        "caption": tf.io.FixedLenFeature([], tf.string),
    }

    # decode the TFRecord
    example = tf.io.parse_single_example(example, feature)

    img = tf.io.decode_raw(example["image_raw"], tf.float32)
    img_tensor = tf.reshape(img, [64, 2048])
    caption = tf.io.decode_raw(example["caption"], tf.int32)

    return img_tensor, caption


numbers = re.compile(r"(\d+)")


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def get_training_dataset(batch_size=BATCH_SIZE, buffered_size=BUFFER_SIZE):
    options = tf.data.Options()
    filenames = sorted(
        tf.io.gfile.glob("gs://tpu-test-koh/tfrecords/*.tfrecord"), key=numericalSort
    )

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


with strategy.scope():
    encoder = I2S_Model.CNN_Encoder(embedding_dim)
    decoder = I2S_Model.RNN_Decoder(embedding_dim, units, vocab_size)

    # Network Parameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00051)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)


checkpoint_path = "gs://tpu-test-koh/checkpoints/train_900k1v3-32_128_TPU_Test_pred"
ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
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


@tf.function
def train_step(iterator):
    def step_fn(inputs):
        img_tensor, target = inputs
        loss = 0

        # initializing the hidden state for each batch because the captions are not related from image to image
        hidden = decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims(
            [tokenizer.word_index["<start>"]] * target.shape[0], 1
        )

        with tf.GradientTape() as tape:
            features = encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = decoder(dec_input, features, hidden)

                loss += loss_function(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = loss / int(target.shape[1])

        trainable_variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        optimizer.apply_gradients(zip(gradients, trainable_variables))

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

    for x in train_dataset:
        img_tensor, target = x
        batch_loss, t_loss = train_step(x)
        total_loss += t_loss
        batch += 1

        if batch % 100 == 0:
            print(
                "Epoch {} Batch {} Loss {:.4f}".format(
                    epoch + 1, batch, batch_loss.numpy() / BATCH_SIZE
                ),
                flush=True,
            )

        if batch == num_steps:
            loss_plot.append(total_loss / num_steps)
            ckpt_manager.save()

            print(
                "Epoch {} Loss {:.6f}".format(epoch + 1, total_loss / num_steps),
                flush=True,
            )
            print(
                datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                "Time taken for 1 epoch {} sec\n".format(time.time() - start),
                flush=True,
            )

            break

plt.plot(loss_plot, "-o", label="Loss value")
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
# plt.show()
plt.gcf().set_size_inches(20, 20)
plt.savefig("Lossplot_900k1v3-32_128_tpu_test.jpg")
plt.close()

print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "Network Completed", flush=True)
f.close()
