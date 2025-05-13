from django.shortcuts import render
import joblib
from django.conf import settings
import os

def index(request):
    prediction=None
    if request.method=="POST":
        msg=request.POST.get("msg")
        model_path_vectroizer=os.path.join(settings.BASE_DIR,"spamdetedtion","model","vectroizer.pkl")
        vectorize=joblib.load(model_path_vectroizer)
        vectorized_msg=vectorize.transform([msg])
        model_path=os.path.join(settings.BASE_DIR,"spamdetedtion","model","spamdetection_file.pkl")
        loaded_model=joblib.load(model_path)
        prediction=loaded_model.predict(vectorized_msg)
        if prediction ==1:
            prediction="spam"
        else:
            prediction="ham" 
    return render (request,"index.html",{'type':prediction})