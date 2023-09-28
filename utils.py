import cv2
import os
import numpy as np
import time

from model.yolo_model import YOLO

def process_image(img):
    """Cambiar el tamaño, reducir y expandir la imagen.

    # Argumento:
        img: imagen original.

    # Retorna
        image: ndarray(64, 64, 3), imagen procesada.
    """
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image

def get_classes(file):
    """Obtener nombres de clases.

    # Argumento:
        file: nombres de clases para la base de datos.

    # Retorna
        class_names: Lista, nombres de las clases.

    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names

def draw(image, boxes, scores, classes, all_classes):
    """Dibujar los cuadros en la imagen.

    # Argumento:
        image: imagen original.
        boxes: ndarray, cuadros de objetos.
        classes: ndarray, clases de objetos.
        scores: ndarray, puntajes de objetos.
        all_classes: nombres de todas las clases.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        print('clase: {0}, puntuación: {1:.2f}'.format(all_classes[cl], score))
        print('coordenadas de la caja x,y,w,h: {0}'.format(box))

    print()

def detect_image(image, yolo, all_classes):
    """Usar YOLO v3 para detectar imágenes.

    # Argumento:
        image: imagen original.
        yolo: YOLO, modelo YOLO.cls
        all_classes: nombres de todas las clases.

    # Retorna:
        image: imagen procesada.
    """
    pimage = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()

    print('tiempo: {0:.2f}s'.format(end - start))

    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)

    return image

def detect_video(video, yolo, all_classes):
    """Usar YOLO v3 para detectar video.

    # Argumento:
        video: archivo de video.
        yolo: YOLO, modelo YOLO.
        all_classes: nombres de todas las clases.
    """
    ruta_del_video = os.path.join("videos", "test", video)
    camara = cv2.VideoCapture(ruta_del_video)
    cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)

    # Preparar para guardar el video detectado
    sz = (int(camara.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camara.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    
    vout = cv2.VideoWriter()
    vout.open(os.path.join("videos", "res", video), fourcc, 20, sz, True)

    while True:
        res, frame = camara.read()

        if not res:
            break

        image = detect_image(frame, yolo, all_classes)
        cv2.imshow("input", image)

        # Guardar el video frame a frame
        vout.write(image)

        if cv2.waitKey(1) & 0xff == 27:
                break

    vout.release()
    camara.release()
    
