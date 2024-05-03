import pickle
import re
import sys
import time
from datetime import datetime

import config
import Efficient_Net_encoder
import efficientnet.tfkeras as efn
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import Transformer_decoder

mpl.use("Agg")

# Set TPUs

tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="node-name")
print("Running on TPU ", tpu.master())

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)

print("Number of devices: {}".format(strategy.num_replicas_in_sync), flush=True)
print("REPLICAS: ", strategy.num_replicas_in_sync, flush=True)
print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "Network Started", flush=True)

# Load the Data
total_data = 1000000  # datasize integer

tokenizer = pickle.load(open("tokenizer_Isomeric_SMILES.pkl", "rb"))
max_length = pickle.load(open("max_length_Isomeric_SMILES.pkl", "rb"))

PAD_TOKEN = tf.constant(tokenizer.word_index["<pad>"], dtype=tf.int32)

# Image parameters
IMG_EMB_DIM = (10, 10, 232)
IMG_EMB_DIM = (IMG_EMB_DIM[0] * IMG_EMB_DIM[1], IMG_EMB_DIM[2])
IMG_SHAPE = (299, 299, 3)
PE_INPUT = IMG_EMB_DIM[0]
IMG_SEQ_LEN, IMG_EMB_DEPTH = IMG_EMB_DIM
D_MODEL = IMG_EMB_DEPTH

# Set Training Epochs
EPOCHS = 40
REPLICA_BATCH_SIZE = 128
BATCH_SIZE = REPLICA_BATCH_SIZE * strategy.num_replicas_in_sync
print(BATCH_SIZE)
BUFFER_SIZE = 10000
target_vocab_size = max_length
TRAIN_STEPS = total_data // BATCH_SIZE
validation_steps = 15360 // BATCH_SIZE

# Parameters to train the network
N_LAYERS = 4
D_MODEL = 512
D_FF = 2048
N_HEADS = 8
DROPOUT_RATE = 0.1

# ENCODER_CONFIG
PREPROCESSING_FN = tf.keras.applications.efficientnet.preprocess_input
BB_FN = Efficient_Net_encoder.get_efficientnetv2_backbone

MAX_LEN = max_length
VOCAB_LEN = len(tokenizer.word_index)
PE_OUTPUT = MAX_LEN
TARGET_V_SIZE = len(tokenizer.word_index)
WARM_STEPS = 4000
TARGET_DTYPE = tf.float32

print("Total train steps: ", TRAIN_STEPS)
AUTO = tf.data.experimental.AUTOTUNE


def decode_image(image_data):
    """Preprocess the input image for Efficient-Net and returned the
    preprocessed image.

    Args:
        image_data (int array): Decoded image in 2D array

    Returns:
        image (array): Preprocessed image in 2D array
    """
    try:
        img = tf.image.decode_png(image_data, channels=3)
    except InvalidArgumentError as e:
        print(e)
        pass
    img = tf.image.resize(img, (299, 299))
    img = efn.preprocess_input(img)
    # img = tf.expand_dims(img, 0)
    # print(img)
    return img


def read_tfrecord(example):
    """Read a tf record file and decodes the image and text data back into
    original form.

    Args:
        example (tf.record): single entry from tf record file

    Returns:
        img (float array): 2D float array
        caption: tokenized SMILES string
    """
    feature = {
        "image_raw": tf.io.FixedLenFeature([], tf.string),
        "caption": tf.io.FixedLenFeature([], tf.string),
    }

    # decode the TFRecord
    example = tf.io.parse_single_example(example, feature)
    img = decode_image(example["image_raw"])
    caption = tf.io.decode_raw(example["caption"], tf.int32)
    return img, caption


numbers = re.compile(r"(\d+)")


def numericalSort(value):
    """Sorts the filenames numerically.

    Args:
        value (int): numerical value of the file name

    Returns:
        parts (int): sorted value
    """
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def get_dataset(batch_size=BATCH_SIZE, buffered_size=BUFFER_SIZE, path=""):
    """Creates a batch of data rom a given dataset.

    Args:
        batch_size (int, optional): number of datapoints per batch. Defaults to BATCH_SIZE.
        buffered_size (int, optional): number of datapoints to be buffered. Defaults to BUFFER_SIZE.
        path (str, optional): path to the location of the dataset. Defaults to "".

    Returns:
        train_dataset (): single batch of datapoints
    """
    options = tf.data.Options()
    # filenames = tf.io.gfile.glob(path)
    filenames = sorted(tf.io.gfile.glob(path), key=numericalSort)
    print(len(filenames))
    fx = open("filenames", "w")
    for i in range(len(filenames)):
        fx.write(filenames[i] + "\n")
    fx.close()

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


train_dataset = strategy.experimental_distribute_dataset(
    get_dataset(path="gs://tpu-test-koh/DECIMER_V2/RanDepict/*.tfrecord")
)
# validation_dataset = strategy.experimental_distribute_dataset(get_dataset(path="gs://tpu-test-koh/DECIMER_V2/RanDepict/Val_data/*.tfrecord"))


# validation_dataset = strategy.experimental_distribute_dataset(get_validation_dataset())

training_config = config.Config()
training_config.initialize_transformer_config(
    vocab_len=VOCAB_LEN,
    max_len=MAX_LEN,
    n_transformer_layers=N_LAYERS,
    transformer_d_dff=D_FF,
    transformer_n_heads=N_HEADS,
    image_embedding_dim=IMG_EMB_DIM,
)
training_config.initialize_encoder_config(
    image_embedding_dim=IMG_EMB_DIM,
    preprocessing_fn=PREPROCESSING_FN,
    backbone_fn=BB_FN,
    image_shape=IMG_SHAPE,
    do_permute=IMG_EMB_DIM[1] < IMG_EMB_DIM[0],
)
training_config.initialize_lr_config(
    warm_steps=WARM_STEPS,
    n_epochs=EPOCHS,
)

print(f"\n Encoder config:\n\t -> {training_config.encoder_config}\n")
print(f"\n Transformer config:\n\t -> {training_config.transformer_config}\n")
print(f"Learning rate config:\n\t -> {training_config.lr_config}\n")

print("Training preparation\n")


def prepare_for_training(lr_config, encoder_config, transformer_config, verbose=0):
    """Preparte the model for training. initiate the learning rate, loss
    object, metrics and optimizer.

    Args:
        lr_config (int): values for learning rate configuration
        encoder_config (_type_): encoder configuration values
        transformer_config (_type_): transformer configuration values
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        loss_function, optimizer, model and the metrics
    """

    with strategy.scope():
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )

        def loss_fn(real, pred):
            # Convert to uint8
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = loss_object(real, pred)
            loss_ *= tf.cast(mask, dtype=loss_.dtype)
            loss_ = tf.nn.compute_average_loss(
                loss_, global_batch_size=REPLICA_BATCH_SIZE
            )
            return loss_

        train_loss = tf.keras.metrics.Mean(name="train_loss", dtype=tf.float32)
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy", dtype=tf.float32
        )
        """validation_loss = tf.keras.metrics.Mean(

        name="validation_loss", dtype=tf.float32 ) validation_accuracy =
        tf.keras.metrics.SparseCategoricalAccuracy(
        name="validation_accuracy", dtype=tf.float32 )
        """
        # Declare the learning rate schedule (try this as actual lr schedule and list...)
        lr_scheduler = config.CustomSchedule(
            transformer_config["d_model"], lr_config["warm_steps"]
        )

        # Instantiate an optimizer
        optimizer = tf.keras.optimizers.Adam(
            lr_scheduler, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )

        # Instantiate the encoder model
        encoder = Efficient_Net_encoder.Encoder(**encoder_config)
        initialization_batch = encoder(
            tf.ones(
                ((REPLICA_BATCH_SIZE,) + encoder_config["image_shape"]),
                dtype=TARGET_DTYPE,
            ),
            training=False,
        )

        # Instantiate the decoder model
        transformer = Transformer_decoder.Transformer(**transformer_config)
        transformer(
            initialization_batch,
            tf.random.uniform((REPLICA_BATCH_SIZE, 1)),
            training=False,
        )

    # Show the model architectures and plot the learning rate
    if verbose:
        print("\nEncoder model\n")
        print(encoder.summary())

        print("\nTransformer model\n")
        print(transformer.summary())

    return (
        loss_fn,
        optimizer,
        lr_scheduler,
        encoder,
        transformer,
        train_loss,
        train_accuracy,
    )


# Instantiate our required training components in the correct scope
(
    loss_fn,
    optimizer,
    lr_scheduler,
    encoder,
    transformer,
    train_loss,
    train_accuracy,
) = prepare_for_training(
    lr_config=training_config.lr_config,
    encoder_config=training_config.encoder_config,
    transformer_config=training_config.transformer_config,
    verbose=1,
)

print("\n Training preparation completed\n")

# Path to checkpoints
checkpoint_path = "gs://tpu-test-koh/DECIMER_V2/checkpoints_SMILES_TPU8/"
ckpt = tf.train.Checkpoint(
    encoder=encoder, transformer=transformer, optimizer=optimizer
)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
    start_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])


# Main training step function
def train_step(image_batch, selfies_batch):
    """Main training step function.

    Args:
        image_batch (float array): Input image batch
        selfies_batch (int array): Input selfies batch tokenized
    """

    selfies_batch_input = selfies_batch[:, :-1]
    selfies_batch_target = selfies_batch[:, 1:]
    combined_mask = Transformer_decoder.create_mask(
        selfies_batch_input, selfies_batch_target
    )

    with tf.GradientTape() as tape:
        image_embedding = encoder(image_batch, training=True)
        prediction_batch, _ = transformer(
            image_embedding,
            selfies_batch_input,
            training=True,
            look_ahead_mask=combined_mask,
        )

        # Update Loss Accumulator
        batch_loss = loss_fn(selfies_batch_target, prediction_batch) / (MAX_LEN - 1)
        train_accuracy.update_state(
            selfies_batch_target,
            prediction_batch,
            sample_weight=tf.where(
                tf.not_equal(selfies_batch_target, PAD_TOKEN), 1.0, 0.0
            ),
        )

    # total_loss = batch_loss / int(selfies_batch.shape[1])

    gradients = tape.gradient(
        batch_loss, encoder.trainable_variables + transformer.trainable_variables
    )
    gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
    optimizer.apply_gradients(
        zip(gradients, encoder.trainable_variables + transformer.trainable_variables)
    )

    train_loss.update_state(batch_loss * strategy.num_replicas_in_sync)


@tf.function
def dist_train_step(image_batch, selfies_batch):
    strategy.run(train_step, args=(image_batch, selfies_batch))


"""
# Main validation step function
def validation_step(image_batch, selfies_batch):

    selfies_batch_input = selfies_batch[:, :-1]
    selfies_batch_target = selfies_batch[:, 1:]
    combined_mask = create_mask(selfies_batch_input, selfies_batch_target)

    with tf.GradientTape() as tape:
        image_embedding = encoder(image_batch, training=True)
        prediction_batch, _ = transformer(
            image_embedding,
            selfies_batch_input,
            training=True,
            look_ahead_mask=combined_mask,
        )

        # Update Loss Accumulator
        batch_loss = loss_fn(selfies_batch_target, prediction_batch) / (MAX_LEN - 1)
        validation_accuracy.update_state(selfies_batch_target, prediction_batch,sample_weight=tf.where(tf.not_equal(selfies_batch_target, PAD_TOKEN), 1.0, 0.0))

    total_loss = batch_loss / int(selfies_batch.shape[1])

    gradients = tape.gradient(
        batch_loss, encoder.trainable_variables + transformer.trainable_variables
    )
    gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
    optimizer.apply_gradients(
        zip(gradients, encoder.trainable_variables + transformer.trainable_variables)
    )

    train_loss.update_state(batch_loss * strategy.num_replicas_in_sync)


@tf.function
def dist_validation_step(image_batch, selfies_batch):
    strategy.run(
        validation_step, args=(image_batch, selfies_batch)
    )
"""
loss_plot = []
accuracy_plot = []
val_loss_plot = []
val_acc = []
f = open("Training_SMILES_TPU8__.txt", "w")
sys.stdout = f

# Training loop
for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    batch = 0
    validation_batch = 0

    for x in train_dataset:
        img_tensor, target = x
        dist_train_step(img_tensor, target)
        batch += 1

        if batch % 100 == 0:
            print(
                "Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()
                ),
                flush=True,
            )
        # storing the epoch end loss value to plot later

        if batch == TRAIN_STEPS:
            loss_plot.append(train_loss.result().numpy())
            accuracy_plot.append(train_accuracy.result().numpy())
            ckpt_manager.save()

            print(
                "Epoch {} Training_Loss {:.4f} Accuracy {:.4f}".format(
                    epoch + 1,
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
    """
    for y in validation_dataset:
        start_val = time.time()
        img_tensor, target =y
        validation_step(y)
        validation_batch +=1

        if validation_batch == validation_steps:

            print ('Validation_Loss {:.4f} Accuracy {:.4f}'.format(validation_loss.result(), validation_accuracy.result()), flush=True)
            print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),'Time taken for validation {} sec\n'.format(time.time() - start_val), flush=True)
            val_loss_plot.append(validation_loss.result().numpy())
            val_acc.append(validation_accuracy.result().numpy())
            break
    """
    train_loss.reset_states()
    train_accuracy.reset_states()
    # validation_loss.reset_states()
    # validation_accuracy.reset_states()

    # epoch = (epoch+1)
    # batch = 0

plt.plot(loss_plot, "-o", label="Training loss")
# plt.plot(val_loss_plot , '-o', label= "Validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(ncol=2, loc="upper right")
# plt.show()
plt.gcf().set_size_inches(20, 20)
plt.savefig("Lossplot_SMILESTPU8.jpg")
plt.close()

plt.plot(accuracy_plot, "-o", label="Training accuracy")
# plt.plot(val_acc , '-o', label= "Validation accuracy")
plt.title("Training and Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(ncol=2, loc="lower right")
# plt.show()
plt.gcf().set_size_inches(20, 20)
plt.savefig("accuracyplot_SMILESTPU8.jpg")
plt.close()

print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "Network Completed", flush=True)
f.close()
