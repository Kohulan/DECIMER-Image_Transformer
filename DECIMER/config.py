import tensorflow as tf

import DECIMER.Efficient_Net_encoder as Efficient_Net_encoder
import DECIMER.Transformer_decoder as Transformer_decoder

TARGET_DTYPE = tf.float32


class Config:
    """Configuration class."""

    def __init__(
        self,
    ):
        self.encoder_config = {}
        self.transformer_config = {}
        self.lr_config = {}

    def initialize_encoder_config(
        self,
        image_embedding_dim,
        preprocessing_fn,
        backbone_fn,
        image_shape,
        do_permute=False,
        pretrained_weights=None,
    ):
        """This functions initializes the Efficient-Net V2 encoder with user
        defined configurations.

        Args:
            image_embedding_dim (int): Embedding dimension of the input image
            preprocessing_fn (method): Efficient Net preprocessing function for input image
            backbone_fn (method): Calls Efficient-Net V2 as backbone for encoder
            image_shape (int): Shape of the input image
            do_permute (bool, optional): . Defaults to False.
            pretrained_weights (keras weights, optional): Use pretrained efficient net weights or not. Defaults to None.
        """
        self.encoder_config = dict(
            image_embedding_dim=image_embedding_dim,
            preprocessing_fn=preprocessing_fn,
            backbone_fn=backbone_fn,
            image_shape=image_shape,
            do_permute=do_permute,
            pretrained_weights=pretrained_weights,
        )

    def initialize_transformer_config(
        self,
        vocab_len,
        max_len,
        n_transformer_layers,
        transformer_d_dff,
        transformer_n_heads,
        image_embedding_dim,
        rate=0.1,
    ):
        """This functions initializes the Transformer model as decoder with
        user defined configurations.

        Args:
            vocab_len (int): Total number of words in the input vocabulary
            max_len (int): Maximum length of the string found on the training dataset
            n_transformer_layers (int): Number of layers present in the transformer model
            transformer_d_dff (int): Transformer feed forward upwards projection size
            transformer_n_heads (int): Number of heads present in the transformer model
            image_embedding_dim (int): Total number of dimension the image gets embedded
            dropout_rate (float, optional): Fraction of the input units to drop. Defaults to 0.1.
        """
        self.transformer_config = dict(
            num_layers=n_transformer_layers,
            d_model=image_embedding_dim,
            num_heads=transformer_n_heads,
            dff=transformer_d_dff,
            target_vocab_size=vocab_len,
            max_len=max_len,
            rate=0.1,
        )

    def initialize_lr_config(self, warm_steps, n_epochs):
        """This function sets the configuration to initialize learning rate.

        Args:
            warm_steps (int): Number of steps The learning rate is increased
            n_epochs (int): Number of epochs
        """
        self.lr_config = dict(
            warm_steps=warm_steps,
            n_epochs=n_epochs,
        )


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom schedule for learning rate used during training.

    Args:
        tf (_type_): keras learning rate schedule
    """

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def prepare_models(encoder_config, transformer_config, replica_batch_size, verbose=0):
    """This function is used to initiate the Encoder and the Transformer with
    appropriate configs set by the user. After initiating the models this
    function returns the Encoder,Transformer and the optimizer.

    Args:
        encoder_config ([type]): Encoder configuration set by user in the config class.
        transformer_config ([type]): Transformer configuration set by user in the config class.
        replica_batch_size ([type]): Per replica batch size set by user(during distributed training).
        verbose (int, optional): Defaults to 0.

    Returns:
        [type]: Optimizer, Encoder model and the Transformer
    """

    # Instantiate an optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00051)

    # Instantiate the encoder model

    encoder = Efficient_Net_encoder.Encoder(**encoder_config)

    # Instantiate the decoder model
    transformer = Transformer_decoder.Decoder(**transformer_config)

    # Show the model architectures and plot the learning rate
    if verbose:
        print("\nEncoder model\n")
        print(encoder.summary())

        print("\nTransformer model\n")
        print(transformer.summary())

    return optimizer, encoder, transformer
