import cv2 
import mediapipe as mp 
import numpy as np 
import time


class DetectorHands: 
    def __init__(self, 
                 mode: bool = False, 
                 number_hands: int = 2, 
                 model_complexity: int = 1, 
                 min_detec_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5): 
        
        # Parametros necessários para inicializar o Hands
        self.mode = mode 
        self.max_num_hands = number_hands 
        self.complexity = model_complexity 
        self.detection_con = min_detec_confidence 
        self.tracking_con = min_tracking_confidence 

        #Inicializando o Hands 
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, 
                                         self.max_num_hands, 
                                         self.complexity, 
                                         self.detection_con,
                                         self.tracking_con)
        
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, 
                    img: np.ndarray, 
                    draw_hands: bool = True): 
        
        #Correção de cor
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Coletar resultados do processo das hands e analisar
        self.results = self.hands.process(img_RGB)
        if self.results.multi_hand_landmarks: 
            for hand in self.results.multi_hand_landmarks:
                if draw_hands: 
                    self.mp_draw.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS)
        
        return img 


if __name__ == '__main__': 
    capture = cv2.VideoCapture(0) 

    Detector = DetectorHands() 
    while True: 
        _, img = capture.read() 

        # Aqui manipularemos o nosso frame
        img = Detector.find_hands(img) #draw_hands=False

        # E retornar o frame com o desenho da mão 

        cv2.imshow("Imagem do Lucas", img)

        if cv2.waitKey(20) & 0xFF==ord('q'): 
            break