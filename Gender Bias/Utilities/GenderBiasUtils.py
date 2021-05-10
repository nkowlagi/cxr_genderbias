import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
tf.compat.v1.enable_eager_execution()
from IPython.display import Image, display
import seaborn as sns
from efficientnet.tfkeras import EfficientNetB3
import numpy as np
import albumentations as A
from keras_retinanet.utils.visualization import draw_box, draw_caption , label_color
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet import models
from PIL import Image
import os
from sklearn import metrics
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.mobilenet_v2 import preprocess_input
from keras import backend as K
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Lambda
import matplotlib.cm as cm
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score
from sklearn.metrics import f1_score

def get_img_array(img_path, size=(300,300)):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def GradCam(img_array, model, last_conv_layer_name, pred_index=None,display_pred=False):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if display_pred:
            print(preds)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
      
    
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img


def get_threshold_shift_pred(predictions,threshold):
    preds=[]
    for pred in predictions:
        if pred[1]>threshold:
            result=1
        else:
            result=0
        #print("Prediction:{}, Result:{}".format(pred,result))    
        preds.append(result)
    return preds  

def get_f1_score(y_true,y_pred,kind):
    return f1_score(y_true,y_pred,average=kind)
    

def print_classification_report(y_true,y_pred):
    cf_matrix=confusion_matrix(np.array(y_true), np.array(y_pred))
    sns.heatmap(cf_matrix, annot=True,cmap='Blues',fmt='g')
    print('Classification Report')
    target_names = ['Not Pneumonia', 'Pneumonia']
    print(classification_report(np.array(y_true), y_pred, target_names=target_names,))

#ROC Plot
def plot_roc(y_true,y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(np.array(y_true),np.array(y_pred))
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
    lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
    

#Plot samples obtained
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

#Pass both Pneumonia 
def save_xrays_roi(xray_folderlist,lsm_model):
    for directory in xray_folderlist:
        for file in os.listdir(directory):
            xray_file=directory+"\\"+file
            imlungs=detect_lungs(xray_file,lsm_model,threshold=1)
            img = Image.fromarray(imlungs)
            img.save("Lungs/"+"Pred_"+file)          
    

def get_prediction_with_lsm(xray_path,lsm_model,lpm_model):
    pathology={0:'No Finding', 1:'Pneumonia'}
    lungs=detect_lungs(xray_path,lsm_model)
    xray=transform_xray(lungs)
    prediction=lpm_model.predict(np.expand_dims(xray, axis=0))
    y_hat = prediction.argmax(axis=-1)
    return pathology[y_hat[0]]

def show_xray(xray_path):
    if isinstance(xray_path, str):
        xray=cv2.imread(xray_path)
        plt.imshow(xray)
    else:
        plt.imshow(xray_path)

#Get model trained on ChexPert
def get_cxp_efficientnetb3():
    model=tf.keras.models.load_model('LPM/CXP_LPM.Epoch36-Val_Acc0.83_Val_Loss0.43.h5')
    return model

#Get model trained on NIHCC dataset
def get_nihcc_efficientnetb3():
    model=tf.keras.models.load_model('LPM/Models/NIHCC_Pneumonia.h5')
    return model

def get_lsm_model():
    model_path='LSM/resnet50_csv_30.h5'
    model = models.load_model(model_path, backbone_name='resnet50')
    model = models.convert_model(model)
    return model

#Resize the image for model input
#Apply clahe to enhance input image
def transform_xray(xray_path):
    transform = A.Compose([A.Resize(300,300)])

    # Read an image with OpenCV and convert it to the RGB colorspace
    if isinstance(xray_path, str):
        xray = cv2.imread(xray_path)
        xray = cv2.cvtColor(xray, cv2.COLOR_BGR2RGB)
    else:
        xray=xray_path
    

    # Augment an image
    transformed = transform(image=xray)
    transformed_xray = transformed["image"]
    return transformed_xray

#Makes the pathology prediction 
#pass the image path and the model
def predict(xray_path,model):
    pathology={0:'No Finding', 1:'Pneumonia'}
    xray=transform_xray(xray_path)
    prediction=model.predict(np.expand_dims(xray, axis=0))
    y_hat = prediction.argmax(axis=-1)
    return pathology[y_hat[0]]

#Get the lungs from the xray
#Pass image path and the model
def detect_lungs(xray_file,model,threshold=1):    
    im = np.array(cv2.imread(xray_file))
        
    #if there's a PNG it will have alpha channel
    im = im[:,:,:3]
    
    imp = preprocess_image(im)
    imp, scale = resize_image(im)
    
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(imp, axis=0))
    
    # standardize box coordinates
    boxes /= scale

    # loop through each prediction for the input image
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < threshold:
            break
            
    box=box.astype(np.int32)
    x0 = box[0]
    y0 = box[1]
    width = box[2]
    height = box[3]
    
    imlungs= im[y0:y0+height , x0:x0+width, :]
    
    return imlungs

#Take image file as input and save the ROI in the desired path
def separate_lungs(img_file,target_path,model,threshold=1):
    im = np.array(cv2.imread(img_file))
    
    # if there's a PNG it will have alpha channel
    im = im[:,:,:3]

    imp = preprocess_image(im)
    imp, scale = resize_image(im)
    
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(imp, axis=0))
    
    # standardize box coordinates
    boxes /= scale

    # loop through each prediction for the input image
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < threshold:
            #print("Image: {}    Score:{:.5f}".format(img_file,score))
            break

    box = box.astype(np.int32)
   
    x = box[0]
    y = box[1]
    width = box[2]
    height = box[3]
    
    lungs = im[y:height , x:width] # Need to revisit based on accuracy of the models

    img = Image.fromarray(lungs)
    img.save(target_path+img_file.replace(".jpg","_Pred.jpg").split("\\")[-1])
    
#Take image file as input and save the ROI in the desired path
def separate_lungsv2(img_file,filename,model,mode='tight',threshold=1):
    im = np.array(cv2.imread(img_file))
    
    # if there's a PNG it will have alpha channel
    im = im[:,:,:3]

    imp = preprocess_image(im)
    imp, scale = resize_image(im)
    
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(imp, axis=0))
    
    # standardize box coordinates
    boxes /= scale

    # loop through each prediction for the input image
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < threshold:
            #print("Image: {}    Score:{:.5f}".format(img_file,score))
            break

    box = box.astype(np.int32)
   
    x = box[0]
    y = box[1]
    width = box[2]
    height = box[3]
    
    if mode=='tight':
        lungs = im[y:height , x:width] # Need to revisit based on accuracy of the models
    else:
        lungs = im[y:y+height , x:x+width]
    
    img = Image.fromarray(lungs)
    img.save(filename)    