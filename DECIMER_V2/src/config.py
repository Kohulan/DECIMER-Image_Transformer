# Network configuration file
import tensorflow as tf
import efficientnet.tfkeras as efn
import Efficient_Net_encoder
import Transformer_decoder

TARGET_DTYPE = tf.float32

def decode_image(image_data):
    img = tf.io.read_file(image_data)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = efn.preprocess_input(img)
    return img

class Config():
    def __init__(self,):
        self.encoder_config = {}
        self.transformer_config = {}
        self.lr_config = {}
    def initialize_encoder_config(self, image_embedding_dim, preprocessing_fn, backbone_fn, image_shape, do_permute=False, pretrained_weights=None):
        self.encoder_config = dict(
            image_embedding_dim=image_embedding_dim, 
            preprocessing_fn=preprocessing_fn, 
            backbone_fn=backbone_fn, 
            image_shape=image_shape, 
            do_permute=do_permute, 
            pretrained_weights=pretrained_weights,
        )
    def initialize_transformer_config(self, vocab_len, max_len, n_transformer_layers, transformer_d_dff, transformer_n_heads, image_embedding_dim, dropout_rate=0.1):
        self.transformer_config = dict(
            num_layers=n_transformer_layers, 
            d_model=image_embedding_dim[-1], 
            num_heads=transformer_n_heads, 
            dff=transformer_d_dff,
            target_vocab_size=vocab_len, 
            pe_input=image_embedding_dim[0], 
            pe_target=max_len, 
            dropout_rate=0.1
        )

def prepare_models(encoder_config, transformer_config, replica_batch_size, verbose=0):

        
    # Instiate an optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00051)
        
    # Instantiate the encoder model 
    
    encoder = Efficient_Net_encoder.Encoder(**encoder_config)
    initialization_batch = encoder(
        tf.ones(((replica_batch_size,)+encoder_config["image_shape"]), dtype=TARGET_DTYPE), 
        training=False,
    )
                
    # Instantiate the decoder model
    transformer = Transformer_decoder.Transformer(**transformer_config)
    transformer(initialization_batch, tf.random.uniform((replica_batch_size, 1)), training=False)
       
    # Show the model architectures and plot the learning rate
    if verbose:
        print("\nEncoder model\n")
        print(encoder.summary())

        print("\nTransformer model\n")
        print(transformer.summary())
  
    return optimizer, encoder, transformer