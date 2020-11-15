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

import speech_recognition as sr
import fasttext.util
from pydub import AudioSegment
from scipy import spatial
import morfeusz2
import plotly.express as px
import numpy as np
import pandas as pd


def hello_world(request):
    context = {}
    if request.method == 'POST':
        fs = FileSystemStorage()
        uploaded_file = request.FILES['upload_file']
        if uploaded_file.name.endswith('.mp3') or uploaded_file.name.endswith('.wav') :
            name = uploaded_file.name
            fs.delete(uploaded_file.name)
            fs.save(uploaded_file.name, uploaded_file)
            if uploaded_file.name.endswith('.mp3'):
                new_name = uploaded_file.name[:-4] + '.wav'
                sound = AudioSegment.from_mp3('media/' + name)
                sound.export('media/' + new_name, format="wav")
                name = new_name
            # wykres, statystyki = audio_to_html('media/' + name)
            # context['audio_wykres'] = wykres
            # context['audio_statystyki'] = statystyki
        elif uploaded_file.name.endswith('.mp4'):
            fs.delete(uploaded_file.name)
            fs.save(uploaded_file.name, uploaded_file)
            html = video_to_html('media/' + uploaded_file.name)
            context['video_wykres'] = html

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


def audio_to_html(file_path):
    myaudio = AudioSegment.from_file(file_path, "wav")  # Nazwa pliku
    n = len(myaudio)
    chunk_length_ms = 3000
    overlap = 1500
    flag = 0
    chunks = []
    for i in range(0, 2 * n, chunk_length_ms):
        if i == 0:
            start = 0
            end = chunk_length_ms
        else:
            start = end - overlap
            end = start + chunk_length_ms
        if end >= n:
            chunks.append(myaudio[start:n])
            break
        chunks.append(myaudio[start:end])
    classes = {"Amusement park": ["park", "zabawa", "rozrywka"],
               "Animals": ["zwierzę", "fauna"],
               "Bench": ["ławka"],
               "Building": ["budynek"],
               "Castle": ["zamek"],
               "Cave": ["jaskinia"],
               "Church": ["kościół"],
               "City": ["miasto", "miejscowość"],
               "Cross": ["krzyż"],
               "Cultural institution": ["kultura", "centrum"],
               "Food": ["jedzenie"],
               "Footpath": ["ścieżka"],
               "Forest": ["las"],
               "Furniture": ["meble"],
               "Grass": ["trawa", "trawnik"],
               "Graveyard": ["cmentarz"],
               "Lake": ["jezioro", "bajoro", "staw"],
               "Landscape": ["krajobraz"],
               "Mine": ["kopalnia"],
               "Monument": ["rzeźba", "statua"],
               "Motor vehicle": ["pojazd", "samochód", "motor"],
               "Mountains": ["góry"],
               "Museum": ["muzeum"],
               "Open-air museum": ["powietrze", "na świeżym powietrzu", "muzeum"],
               "Park": ["park"],
               "Person": ["osoba", "człowiek", "ludzie"],
               "Plants": ["roślinność", "flora", "roślina"],
               "Reservoir": ["rezerwat"],
               "River": ["rzeka", "strumień"],
               "Road": ["droga"],
               "Rocks": ["kamień", "skała"],
               "Snow": ["śnieg"],
               "Sport": ["sport"],
               "Sports facility": ["ośrodek sportowy"],
               "Stairs": ["schody"],
               "Trees": ["drzewo"],
               "Watercraft": ["łódź", "łódka"],
               "Windows": ["okno"]}

    ft = fasttext.load_model('media/embeddings.bin')

    classes_vec = {}
    for label in classes:
        classes_vec[label] = []
        for i in range(len(classes[label])):
            classes_vec[label].append(ft.get_word_vector(classes[label][i]))
    r = sr.Recognizer()
    morf = morfeusz2.Morfeusz()
    data = []
    for moment, chunk in enumerate(chunks):
        chunk.export("chunk.wav", format="wav")
        a = sr.AudioFile("chunk.wav")
        with a as source:
            audio = r.record(source)
        try:
            text = r.recognize_google(audio, language='pl-PL')
        except:
            continue

        analyse = morf.analyse(text)
        words = []
        for (i, j, (orth, base, tag, posp, kwal)) in analyse:
            index = base.find(":")
            if index > -1:
                words.append(base[:index])
            else:
                words.append(base)
        words = list(set(words))
        for word in words:
            word_vec = ft.get_word_vector(word)
            for label in classes_vec:
                similarity = 0
                for i in range(len(classes_vec[label])):
                    similarity += 1 - spatial.distance.cosine(word_vec, classes_vec[label][i])
                similarity = similarity / len(classes_vec[label])
                if similarity > 0.6:
                    data.append([label, (moment + 1) * 0.5 * chunk_length_ms / 1000])

    data2 = []
    for i in range(len(data) - 1):
        if (data[i][0] == data[i + 1][0]) and (data[i][1] + 1.5 == data[i + 1][1]):
            x = [data[i][0], (data[i][1] + data[i + 1][1]) / 2]
            data2.append(x)
        else:
            data2.append(data[i])
            if i == len(data) - 2:
                data2.append(data[i + 1])

    df = pd.DataFrame(data2)
    df.iloc[:, 0] = df.iloc[:, 0].astype("category")
    df[3] = df.iloc[:, 0].cat.codes

    fig = px.scatter(x=df.iloc[:, 1], y=df.iloc[:, 2], hover_name=df.iloc[:, 0], color=df.iloc[:, 0])
    fig.update_layout(title_text="", showlegend=False)
    fig.update_yaxes(title_text='Labels', ticktext=df.iloc[:, 0], tickvals=df.iloc[:, 2], showgrid=True, zeroline=False,
                     fixedrange=True)
    fig.update_xaxes(title_text='Time [s]', nticks=80)
    wykresik = plotly.io.to_html(fig)

    fig2 = px.histogram(x=df.iloc[:, 0])
    fig2.update_layout(title_text="")
    fig2.update_xaxes(title_text='Labels')
    statystyki = plotly.io.to_html(fig2)

    return wykresik, statystyki
