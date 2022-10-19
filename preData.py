import tensorflow as tf
import os
AUTOTUNE=tf.data.experimental.AUTOTUNE

def createTFREC(path,name):
    image_path=os.listdir(path)
    for i in range(len(image_path)):
        image_path[i]=path+image_path[i]
    ds=tf.data.Dataset.from_tensor_slices(image_path).map(tf.io.read_file)
    tfrec=tf.data.experimental.TFRecordWriter(name)
    tfrec.write(ds)

createTFREC('./data/train/gray/','./trainA.tfrec')
createTFREC('./data/train/ori/','./trainB.tfrec')
createTFREC('./data/test/gray/','./testA.tfrec')
createTFREC('./data/test/ori/','./testB.tfrec')
createTFREC('./data/validation/gray/','./validationA.tfrec')
createTFREC('./data/validation/ori/','./valiationB.tfrec')
# image_ds=tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.io.read_file)
# tfrec=tf.data.experimental.TFRecordWriter('trainA.tfrec')
# tfrec.write(image_ds)


# image_ds=tf.data.TFRecordDataset('trainA.tfrec').map(preprocess_image)
