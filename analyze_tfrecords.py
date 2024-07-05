import tensorflow as tf

def print_tfrecords(file_path, num_records=10):
    # Create a TFRecordDataset
    raw_dataset = tf.data.TFRecordDataset(file_path)

    # Iterate over the dataset and print the first few records
    for idx, raw_record in enumerate(raw_dataset.take(num_records)):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        print(f"Record {idx + 1}:")
        for key, feature in example.features.feature.items():
            kind = feature.WhichOneof('kind')
            if kind == 'bytes_list':
                value = feature.bytes_list.value
            elif kind == 'float_list':
                value = feature.float_list.value
            elif kind == 'int64_list':
                value = feature.int64_list.value
            print(f"  {key}: {value}")

# Example usage
print_tfrecords('/media/data_cifs/clicktionary/clickme_experiment/tf_records/archive/clickme_train.tfrecords')



