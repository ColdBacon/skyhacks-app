import os

import cv2
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import pytorch_lightning as pl
from PIL import Image
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import Linear, BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from torchvision import transforms
from tqdm import tqdm


def hello_world(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['upload_file']
        fs = FileSystemStorage()
        fs.delete(uploaded_file.name)
        fs.save(uploaded_file.name, uploaded_file)
        print('start')
        print(os.getcwd())
        html = video_to_html('media/' + uploaded_file.name)
        context['wykres'] = html
    # if request.method == 'POST':
    #     uploaded_file = request.FILES['upload_file']
    #     print("nazwa pliku: ", uploaded_file.name)
    #     if not uploaded_file.name.endswith('.mp3') or not uploaded_file.name.endswith('.mp4'):
    #         error = True
    #         context['error'] = error
    #         messages.error(request, 'Please upload a .mp3 file.')
    #         print('Please upload a .mp3 or .mp4 file.')
    #     elif uploaded_file.name.endswith('.mp3'):
    #         fs = FileSystemStorage()
    #         fs.delete('file.mp3')
    #         name = fs.save('file.mp3', uploaded_file)
    #         #url = fs.url(name)
    #         os.system('python -m nbconvert --to html --TemplateExporter.exclude_input=True --execute ./audio.ipynb')
    #         #data = "python ./world.py " + url
    #         #os.system(data)
    #     elif uploaded_file.name.endswith('.mp4'):
    #         f
    #     else:
    #         print("hehe... xD")

    # exec(open("./world.py").read())

    # form = Files(request.POST or None)
    # context['form'] = form
    # return render(request, "upload.html", context)
    return render(request, "upload.html", context)


class SkyhacksModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resnet = models.resnext50_32x4d(pretrained=True)
        self.resnet.fc = Linear(2048, 38)
        self.sigmoid = nn.Sigmoid()
        self.loss = BCEWithLogitsLoss()

    def forward(self, x):
        x = self.resnet(x)  # Sprawdzyci czy tu softmaxa nie trzeba walnac// edit chyba nie czeba bo loss to ma
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss(output.float(), y.float())
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss(output.float(), y.float())
        losses = float(loss.cpu())
        f1 = f1_score(np.array(self.sigmoid(output).cpu() > 0.5) * 1, np.array(y.cpu()), average='macro')
        f1_micro = f1_score(np.array(self.sigmoid(output).cpu() > 0.5) * 1, np.array(y.cpu()), average='micro')
        return losses, f1, f1_micro

    def validation_epoch_end(self, val_step_outputs):
        losses, f1, f1_micro = 0, 0, 0
        for l, f, f_mic in val_step_outputs:
            losses += l
            f1 += f
            f1_micro += f_mic

        f1 = f1 / len(val_step_outputs)
        f1_micro = f1_micro / len(val_step_outputs)
        valid_loss = losses / len(val_step_outputs)
        self.log('val_f1', f1, on_epoch=True, prog_bar=True, logger=True)
        self.log('f1_micro', f1_micro, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', valid_loss, on_epoch=True, prog_bar=True, logger=True)
        print('val_f1, f1_micro, val_loss', f1, f1_micro, valid_loss)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = StepLR(optimizer, step_size=7)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


def video_to_dataframe(video_path, model, preprocess, columns):
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    success = True
    print(type(vidcap))
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    print('fps', fps)

    data = list()
    while success:
        success, image = vidcap.read()
        if not success:
            break
        if count % int(1 * fps) == 0:  # 1 oznacza co ile sekund
            td = int(count / fps)
            im_pil = Image.fromarray(image).convert('RGB')
            data.append([td, im_pil])
        count += 1

    vidcap.release()

    result_csv = list()
    for td, img in tqdm(data):
        x = preprocess(img)
        output = model(x.unsqueeze(0))
        logits = (model.sigmoid(output) > 0.5) * 1
        result_csv.append([td, *logits[0].tolist()])
    df = pd.DataFrame(result_csv, columns=columns)
    return df


def video_to_html(video):
    model = SkyhacksModel.load_from_checkpoint('media/wagi.ckpt')  # model directory
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    row = 'Timedelta,Amusement park,Animals,Bench,Building,Castle,Cave,Church,City,Cross,Cultural institution,Food,Footpath,Forest,Furniture,Grass,Graveyard,Lake,Landscape,Mine,Monument,Motor vehicle,Mountains,Museum,Open-air museum,Park,Person,Plants,Reservoir,River,Road,Rocks,Snow,Sport,Sports facility,Stairs,Trees,Watercraft,Windows'
    col_names = [str(i) for i in row.split(',')]
    df = video_to_dataframe(video, model, preprocess=preprocess, columns=col_names)

    df = df.drop(df.columns[1], axis=1)

    data = []
    for index, row in df.iterrows():
        for col in df.columns[1:]:
            if row[col] == 1:
                data.append([col, row[0]])

    df = pd.DataFrame(data)
    df.iloc[:, 0] = df.iloc[:, 0].astype("category")
    df[3] = df.iloc[:, 0].cat.codes

    fig = px.scatter(x=df.iloc[:, 1], y=df.iloc[:, 2], hover_name=df.iloc[:, 0], color=df.iloc[:, 0])
    fig.update_layout(title_text="", width=800, height=40 * (max(df.iloc[:, 2]) + 1), showlegend=False)
    fig.update_yaxes(title_text='Labels', ticktext=df.iloc[:, 0], tickvals=df.iloc[:, 2], showgrid=True, zeroline=False,
                     fixedrange=True)
    fig.update_xaxes(title_text='Time [s]', nticks=100)
    result = plotly.io.to_html(fig)
    return result
