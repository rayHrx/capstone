import tensorflow as tf

def model_fn(features,labels,mode,params): 
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    conv1 = tf.layer.conv2d(
        inputs=input_layer,
        filter=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layer.max_pooling2d(inputs=conv1, pool_size=[2, 2], stride=2)
    pool1_flat = tf.reshape(pool1, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout, params['n_classes'], activation=None)
    
    predictions = {
        "classes" : tf.argmax(input=logits, axis=1),
        "probability": tf.nn.softmax(logits, name="softmax_tensor")
    }
    # Predict mode
    if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logtis)
    # Train mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    # Eval mode
    eval_metric_ops = {
        "accuracy" : tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
        }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)