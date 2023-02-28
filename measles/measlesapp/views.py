from django.shortcuts import render
from django.core.files.storage import default_storage
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
# Create your views here.
def welcome (request):
    if request.method =="POST":
        file =request.FILES["image"]
        file_name = default_storage.save(file.name,file)
        imp_predr= default_storage.path(file_name)
                        
        models = load_model('mlmodels/model.h5py')
        #imp_pred =image.load_img('../dataset/measles/train/normal/normal16.png' )
        img = load_img(imp_predr,target_size =(224,224))
        imp_pred = img_to_array(img)

        imp_pred =imp_pred/255
        imp_pred = imp_pred.reshape(1,224,224,3)
        result = models.predict(imp_pred)
        
        classes =['measles','monkeypox','normal','other']
        result = np.argmax(result)
        result = classes[result]
        print(result)
        return render(request,'measleshome/home.html' ,{'prediction':result,'imager':imp_predr})



    else:
        return render(request,'measleshome/home.html')
    return render(request,'measleshome/home.html')
    

def about(request):
    return render(request,'measleshome/about.html')




