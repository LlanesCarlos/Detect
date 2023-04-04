import os
import logging
import time
import cv2
import numpy as np
from keras.models import load_model
from logging.handlers import TimedRotatingFileHandler


cara = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
ojo_izq = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
ojo_der = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl = ['Cerrar', 'Abrir']

modelo = load_model('models/modelo.h5')
direc = os.getcwd()
captura = cv2.VideoCapture(0)
fuente = cv2.FONT_HERSHEY_COMPLEX_SMALL
cuenta = 0
puntaje = 0
grosor = 2
pred_der = [99]
pred_izq = [99]

logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)

handler = TimedRotatingFileHandler('log.txt', when='s', interval=5, backupCount=10)
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

while True:
    ret, frame = captura.read()
    altura, ancho = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    caras = cara.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    ojo_izquierdo = ojo_izq.detectMultiScale(gray)
    ojo_derecho = ojo_der.detectMultiScale(gray)

    cv2.rectangle(frame, (0, altura - 50), (300, altura), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in caras:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in ojo_derecho:
        o_der = frame[y:y + h, x:x + w]
        cuenta = cuenta + 1
        o_der = cv2.cvtColor(o_der, cv2.COLOR_BGR2GRAY)
        o_der = cv2.resize(o_der, (24, 24))
        o_der = o_der / 255
        o_der = o_der.reshape(24, 24, -1)
        o_der = np.expand_dims(o_der, axis=0)
        pred_der = np.argmax(modelo.predict(o_der), axis=-1)
        if pred_der[0] == 1:
            lbl = 'Abierto '
        if pred_der[0] == 0:
            lbl = 'Cerrado '
        break

    for (x, y, w, h) in ojo_izquierdo:
        o_izq = frame[y:y + h, x:x + w]
        cuenta = cuenta + 1
        o_izq = cv2.cvtColor(o_izq, cv2.COLOR_BGR2GRAY)
        o_izq = cv2.resize(o_izq, (24, 24))
        o_izq = o_izq / 255
        o_izq = o_izq.reshape(24, 24, -1)
        o_izq = np.expand_dims(o_izq, axis=0)
        pred_izq = np.argmax(modelo.predict(o_izq), axis=-1)
        if pred_izq[0] == 1:
            lbl = 'Abiertos'
        if pred_izq[0] == 0:
            lbl = 'Cerrados'
        break

    if pred_der[0] == 0 and pred_izq[0] == 0:
        puntaje = puntaje + 1
        cv2.putText(frame, "Cerrados ", (10, altura - 20), fuente, 1, (255, 255, 255), 1, cv2.LINE_AA)
    # if pred_der[0]==1 or pred_izq[0]==1:
    else:
        puntaje = puntaje - 1
        cv2.putText(frame, "Abiertos ", (10, altura - 20), fuente, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if puntaje < 0:
        puntaje = 0
    cv2.putText(frame, 'Puntaje: ' + str(puntaje), (150, altura - 20), fuente, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if puntaje >= 15:
        puntaje = 15

        if grosor < 16:
            grosor = grosor + 2
        else:
            grosor = grosor - 2
            if grosor < 2:
                grosor = 2
        cv2.rectangle(frame, (0, 0), (ancho, altura), (0, 0, 255), grosor)

    logger.debug(puntaje)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()
