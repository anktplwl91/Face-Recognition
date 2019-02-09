# Face-Recognition
Face Detection using open-cv and Face Recognition using Siamese networks using Keras and TensorFlow

This project can be used to train a Siamese network for Face Recognition based on either Contrastive Loss and Triplet Loss as well.
I have implemented this project using Keras (with TensorFlow backend) and open-cv in Python 3.6.

The complete pipeline for training the network is as follows:

1. Extract faces from human images, which are stored in separate person-wise directories. You can use **face_detection.py** for
this purpose. This python script takes two inputs,

    (i) *-n/--new_image-dir* which stores the faces of all people in seperate person-wise directories, and 
    
    (ii) *-o/--orig_image_dir* which is a list of original directory names. Keep in mind that all directories mentioned after -o 
    flag must contain images of all people in seperate folders. Sample call is like

                      python face_recognition.py -n face_images -o people_dir_1 people_dir_2

2. Having face images stored in the directory mentioned above, you can now run either *face_recog_contrastive.py* or 
*face_recog_triplet.py* based on wheether Contrastive Loss or Triplet Loss is to be used to train the network. Both python scripts
takes same inputs, namely

    (i) *-b/--batch_size* represents the batch-size of inputs to be given to network, it defaults to 16.
  
    (ii) *-d/--img_dir* represents directory which was mentioned as target directory above containing face images.
  
    (iii) *-m/--model* represents the TensorFlow model which is to be used as base model for Siamese networks. Allowed values are -
    mobilenet(MobileNetV2), inception(InceptionV3), vgg(VGG19), xception(Xception), resnet(ResNet50) and inception-resnet
    (Inception-ResNet), defaults to mobilenet.
  
    (iv) *-o/--optim* represents the optimizer to be used to train the network. Allowed values are sgd (default), rmsprop and adam. All
    optimizers are used with default parameters.
  
    (v) *-e/--epochs* represents the number of epochs for which the model is to be trained, defaults to 5.
    
    Sample call is like
    
                    python face_recog_contrastive.py -b 32 -d face_dirs -m resnet -o rmsprop -e 20

