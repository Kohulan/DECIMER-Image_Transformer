# EfficientNet-V2 config
import tensorflow as tf

import DECIMER.efficientnetv2
from DECIMER.efficientnetv2 import effnetv2_configs
from DECIMER.efficientnetv2 import effnetv2_model

BATCH_SIZE_DEBUG = 2
MODEL = "efficientnetv2-m"  # @param


# Define encoder
def get_efficientnetv2_backbone(
    model_name, include_top=False, input_shape=(512, 512, 3), pooling=None, weights=None
):
    # Catch unsupported arguments
    if pooling or weights or include_top:
        raise NotImplementedError(
            "\n...At this time we only want to use the raw "
            "(no pretraining), headless, features with no pooling ...\n"
        )
    backbone = effnetv2_model.EffNetV2Model(model_name=model_name)
    backbone(
        tf.ones((BATCH_SIZE_DEBUG, *input_shape)), training=False, features_only=True
    )
    return backbone


class Encoder(tf.keras.Model):
    def __init__(
        self,
        image_embedding_dim,
        preprocessing_fn,
        backbone_fn,
        image_shape,
        do_permute=False,
        include_top=False,
        pretrained_weights=None,
        scale_factor=0,
    ):
        super(Encoder, self).__init__()

        self.image_embedding_dim = image_embedding_dim
        self.preprocessing_fn = preprocessing_fn
        self.encoder_backbone = backbone_fn(
            model_name=MODEL,
            include_top=include_top,
            weights=pretrained_weights,
            input_shape=image_shape,
        )
        self.reshape = tf.keras.layers.Reshape(
            self.image_embedding_dim, name="image_embedding"
        )
        self.include_top = include_top
        self.scale_factor = scale_factor

    def call(self, x, training):
        x = self.preprocessing_fn(x)
        x = self.encoder_backbone(
            x, training=training, features_only=not self.include_top
        )[self.scale_factor]
        x = self.reshape(x, training=training)
        return x
