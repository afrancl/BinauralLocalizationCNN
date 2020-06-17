import tensorflow as tf

def record_tensor_mean(tensor_in,update_collection="metrics_update",
                        metrics_collection="metrics_out"):
    mean_value,mean_update = tf.metrics.mean_tensor(tensor_in,
                                                    metrics_collections=metrics_collection,
                                                    updates_collections=update_collection)
    return mean_value,mean_update
