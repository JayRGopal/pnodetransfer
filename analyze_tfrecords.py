import tensorflow as tf

TFRECORDS_PATH = '/media/data_cifs/clicktionary/clickme_experiment/tf_records/archive/clickme_train.tfrecords'


def list_tfrecords_features(file_path, num_records=10):
    # Create a TFRecordDataset
    raw_dataset = tf.data.TFRecordDataset(file_path)
    
    feature_names = set()
    
    # Iterate over the dataset and collect feature names from the first few records
    for idx, raw_record in enumerate(raw_dataset.take(num_records)):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        for key in example.features.feature.keys():
            feature_names.add(key)
    
    print("Features found in the TFRecords file:")
    for feature in feature_names:
        print(f"  {feature}")

# Example usage
list_tfrecords_features(TFRECORDS_PATH)



def count_label_values(file_path, label_name='label', num_records=10):
    # Create a TFRecordDataset
    raw_dataset = tf.data.TFRecordDataset(file_path)
    
    label_counts = {}

    # Iterate over the dataset and collect label values
    for idx, raw_record in enumerate(raw_dataset.take(num_records)):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        if label_name in example.features.feature:
            feature = example.features.feature[label_name]
            kind = feature.WhichOneof('kind')
            
            if kind == 'bytes_list':
                values = feature.bytes_list.value
            elif kind == 'float_list':
                values = feature.float_list.value
            elif kind == 'int64_list':
                values = feature.int64_list.value

            for value in values:
                label_counts[value] = label_counts.get(value, 0) + 1
    
    print(f"Counts for label '{label_name}':")
    total_count = 0
    for value, count in label_counts.items():
        print(f"  {value}: {count}")
        total_count += count
      
    print(f'Total Count: {total_count}')

# Example usage
count_label_values(TFRECORDS_PATH, label_name='label', num_records=1000)


