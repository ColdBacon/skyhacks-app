from django.shortcuts import render
from .forms import Files
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
import os


def hello_world(request):

    context = {} 
    error = False

    if request.method == 'POST':
        uploaded_file = request.FILES['upload_file']
        print("nazwa pliku: ", uploaded_file.name)
        if not uploaded_file.name.endswith('.mp3') or not uploaded_file.name.endswith('.mp3') :
            error = True
            context['error'] = error
            messages.error(request, 'Please upload a .mp3 file.')
            print('Please upload a .mp3 or .mp4 file.')
        elif uploaded_file.name.endswith('.mp3'):
            fs = FileSystemStorage()
            fs.delete('file.mp3')
            name = fs.save('file.mp3', uploaded_file)
            #url = fs.url(name)
            os.system('python -m nbconvert --to html --TemplateExporter.exclude_input=True --execute ./audio.ipynb')
            #data = "python ./world.py " + url
            #os.system(data)
        elif uploaded_file.name.endswith('.mp4'):
            fs = FileSystemStorage()
            fs.delete('file.mp4')
            name = fs.save('file.mp4', uploaded_file)
            os.system('jupyter nbconvert --to html --TemplateExporter.exclude_input=True --execute ./video.ipynb')
        else:
            print("hehe... xD")


        #exec(open("./world.py").read())

    #form = Files(request.POST or None) 
    #context['form'] = form 
    #return render(request, "upload.html", context)
    return render(request, "upload.html", context)