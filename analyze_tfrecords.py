import tensorflow as tf

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
print_tfrecords('/media/data_cifs/clicktionary/clickme_experiment/tf_records/archive/clickme_train.tfrecords')



