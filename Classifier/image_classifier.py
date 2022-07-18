import tensorflow as tf
from tensorflow.keras import  layers, models
import os
from pathlib import Path
import argparse


#__file__ = os.getcwd()
data_dir = Path(__file__).resolve().parents[1] / "Data" / "inaturalist_12K" / "Train"

batch_size = 32
img_height = 400
img_width = 400

def prepare_data():

  train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


  val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)


  class_names = train_ds.class_names


  AUTOTUNE = tf.data.AUTOTUNE

  train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

  return train_ds, val_ds, class_names






def model_creation(args, img_height, img_width, nr_classes):

  model = models.Sequential()
  model.add(layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)))

  nr_cnn_layer = 5
  count = 0

  while count < nr_cnn_layer:
      nr_filter = args.cnn_nos[count]
      size_filter = args.cnn_size[count]
      filter_activation = args.act_fn
      model.add(layers.Conv2D(nr_filter, (size_filter,size_filter), activation= filter_activation))
      model.add(layers.MaxPooling2D((2, 2)))
      count +=1

  model.add(layers.Flatten())
  model.add(layers.Dense(args.fc_size, activation=args.act_fn))
  model.add(layers.Dense(nr_classes))
  
  return model

def run_experiment(args, train_ds, val_ds, class_names):

  # cnn_param = [nr of filter, size of filter, activation]
  #cnn_params = [[32,3,'relu'],[32,3,'relu'],[32,3,'relu'],[32,3,'relu'],[32,3,'relu']]
  # ff_param = [nr of neurons, activation]
  #ff_params = [256,'relu']
  #input_size  = (img_height, img_width)

  nr_classes = len(class_names)
  model = model_creation(args,img_height, img_width, nr_classes)


  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])


  epochs=10
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
  )
  
 




def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn_nos", nargs=5, type=int, help=" Provide number of filters for each CNN layer.")
    parser.add_argument("--act_fn", type=str, help=" Provide activation function.")
    parser.add_argument("--cnn_size", nargs=5, type=int, help=" Provide filter size for each CNN layer.")
    parser.add_argument("--fc_size", type=int, help=" Provide size of fully connected layer.")
    

    args = parser.parse_args()
    return args


def main():
    """Run experiment."""

    args = _parse_args()
   
    train_ds, val_ds, class_names = prepare_data()
   
    run_experiment(args, train_ds, val_ds, class_names)


if __name__ == "__main__":
    main()