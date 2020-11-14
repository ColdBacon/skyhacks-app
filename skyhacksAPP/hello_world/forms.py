from django import forms 
  
class Files(forms.Form): 
    file_mp3 = forms.FileField(required=False)
    file_mp4 = forms.FileField(required=False)

#upload_to = 'audio/'

    
