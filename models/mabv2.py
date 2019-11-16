import tensorflow as tf
from attention import *

def layer_process(x, mode):
    if not mode or mode == "none":
        return x
    elif mode == "layer_norm":
        return layer_norm(x)
    else:
        raise ValueError("Unknown mode %s" % mode)


def residual_fn(x, y, dropout):
    y = tf.layers.dropout(y, dropout)
    return x + y

def MAB(visual, language, bias, params, scope=None):
    with tf.variable_scope(scope, default_name="mab"):
        for layer in range(params.num_mab):
            with tf.variable_scope("layer_%d" % layer):
                x = language

                with tf.variable_scope("self_attention_language"):
                    y = multihead_attention(
                        layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.hidden_size,
                        params.hidden_size,
                        params.hidden_size,
                        params.attention_dropout
                    )
                    y = y["outputs"]
                    x = residual_fn(x, y, params.residual_dropout)
                    x = layer_process(x, params.layer_postprocess)

                with tf.variable_scope("multi_attention_language"):
                    y = multihead_attention(
                        layer_process(x, params.layer_preprocess),
                        visual,
                        None,
                        params.num_heads,
                        params.hidden_size,
                        params.hidden_size,
                        params.hidden_size,
                        params.attention_dropout
                    )
                    y = y["outputs"]
                    x = residual_fn(x, y, params.residual_dropout)
                    x = layer_process(x, params.layer_postprocess)
                        
                with tf.variable_scope("feed_forward_language"):
                    y = ffn_layer(
                        layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        params.relu_dropout,
                    )
                    x = residual_fn(x, y, params.residual_dropout)
                    x = layer_process(x, params.layer_postprocess)

                language = x

                x = visual

                with tf.variable_scope("self_attention_visual"):
                    y = multihead_attention(
                        layer_process(x, params.layer_preprocess),
                        None,
                        None,
                        params.num_heads,
                        params.hidden_size,
                        params.hidden_size,
                        params.hidden_size,
                        params.attention_dropout
                    )
                    y = y["outputs"]
                    x = residual_fn(x, y, params.residual_dropout)
                    x = layer_process(x, params.layer_postprocess)
                
                with tf.variable_scope("multi_attention_visual"):
                    y = multihead_attention(
                        layer_process(x, params.layer_preprocess),
                        language,
                        bias,
                        params.num_heads,
                        params.hidden_size,
                        params.hidden_size,
                        params.hidden_size,
                        params.attention_dropout
                    )
                    y = y["outputs"]
                    x = residual_fn(x, y, params.residual_dropout)
                    x = layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward_visual"):
                    y = ffn_layer(
                        layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        params.relu_dropout,
                    )
                    x = residual_fn(x, y, params.residual_dropout)
                    x = layer_process(x, params.layer_postprocess)

                visual = x
        return x