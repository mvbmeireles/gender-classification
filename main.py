# fonte: https://www.thepythoncode.com/article/gender-detection-using-opencv-in-python
# Importar bibliotecas
import cv2
import numpy as np
import os
import uuid

# Arquitetura do modelo para classificação de gênero
# https://drive.google.com/open?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ
GENDER_MODEL = 'weights/deploy_gender.prototxt'
# Os pesos para o modelo pré-treinado
# https://drive.google.com/open?id=1AW3WduLk1haTVAxHOkVS_BEzel1WXQHP
GENDER_PROTO = 'weights/gender_net.caffemodel'
# Cada modelo Caffe impõe a forma da imagem de entrada e o pré-processamento da imagem é necessário,
# como subtração da média, para eliminar o efeito das mudanças de iluminação
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# Representação das classes de genêros
GENDER_LIST = ['Male', 'Female']
# https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
FACE_PROTO = "weights/deploy.prototxt"
# https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# Carregar modelo Caffe
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
# Carregar modelo de predição de gênero
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

def get_faces(frame, confidence_threshold=0.5):
    """Função responsável por encontrar os rostos em uma imagem"""
    # Converter o frame em blob para ser utilizado no NN
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    # Utilizar a imagem como input para o NN
    face_net.setInput(blob)
    # Performar inferência e previsões
    output = np.squeeze(face_net.forward())
    # Lista vazia para adicionar os resultados
    faces = []
    # Loop nas faces detectadas
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * \
                np.array([frame.shape[1], frame.shape[0],
                         frame.shape[1], frame.shape[0]])
            # Converter limites dos retângulos das faces encontradas em inteiros
            start_x, start_y, end_x, end_y = box.astype(np.int)
            # Ampliar um pouco o retângulo
            start_x, start_y, end_x, end_y = start_x - \
                10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            # Incluir na lista
            faces.append((start_x, start_y, end_x, end_y))
    return faces

# fonte: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    """Função responsável por redimensionar os tamanhos da imagem"""
    # Inicializar as dimensões para redimensionar a imagem e obter o tamanho da imagem original
    dim = None
    (h, w) = image.shape[:2]
    # Se largura e altura forem None, retornar a imagegm com as dimensões originais
    if width is None and height is None:
        return image
    # Checar se a largura é None
    if width is None:
        # Calcular a razão entre as alturas desejada e original, e construir a variável de dimensões
        r = height / float(h)
        dim = (int(w * r), height)
    # Caso contrário, a altura será None
    else:
        # Calcular a razão entre as larguras desejada e original, e construir a variável de dimensões
        r = width / float(w)
        dim = (width, int(h * r))
    # Redimensionar a imagem
    return cv2.resize(image, dim, interpolation = inter)

def predict_gender(input_path: str):
    """Função responsável por prever os genêros das faces detectadas na imagem"""
    # Ler imagem
    img = cv2.imread(input_path)
    # Descomentar caso desejar redimensionar a imagem
    # img = cv2.resize(img, (frame_width, frame_height))
    # Fazer uma cópia da imagem inicial e redimensionar a cópia
    frame = img.copy()
    #if frame.shape[1] > frame_width:
    #    frame = image_resize(frame, width=frame_width)
    # Prever as faces na imagem
    faces = get_faces(frame)
    # Loop nas imagens detectadas
    for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
        face_img = frame[start_y: end_y, start_x: end_x]
        # image --> Imagem a ser pré-processada antes de ser passada para o CNN
        # scale factor = Após realizar a subtração média, podemos opcionalmente redimensionar a imagem por algum fator. (se 1 -> sem escala)
        # size = O tamanho espacial que o CNN espera. As opções são = (224*224, 227*227 ou 299*299)
        # mean = valores médios de subtração a serem subtraídos de cada canal da imagem.
        # swapRB=OpenCV espera imagens em BGR enquanto a média é fornecida em RGB. Para resolver isso, definimos swapRB como True.
        blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=1.0, size=(
            227, 227), mean=MODEL_MEAN_VALUES, swapRB=False, crop=False)
        # Prediz o gênero
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        i = gender_preds[0].argmax()
        gender = GENDER_LIST[i]
        # Escolhe um nome aleatório para a imagem a ser salva
        filename = str(uuid.uuid4())
        # Salva as imagens em pastas separadas em feminino e masculino, de acordo com o gênero previsto
        if gender == "Female":
            if not os.path.exists("./new_dataset/feminino"):
                os.mkdir("./new_dataset/feminino")
            cv2.imwrite("./new_dataset/feminino/{}.jpg".format(filename), face_img)
        elif gender == "Male":
            if not os.path.exists("./new_dataset/masculino"):
                os.mkdir("./new_dataset/masculino")
            cv2.imwrite("./new_dataset/masculino/{}.jpg".format(filename), face_img)

if __name__ == '__main__':
    # Parsing command line arguments entered by user
    predict_gender(".\dataset\QgqFDU7RfxtFx1AL.png")

