import cv2
import mediapipe as mp
 
# Inicializa os módulos do MediaPipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
 
# Insira o caminho do seu vídeo aqui
caminho_video = 'safepoint/video.mp4'  # Exemplo: 'C:/meus_videos/video.mp4'
 
# Inicializa os detectores
hands = mp_hands.Hands()
face_detection = mp_face.FaceDetection(min_detection_confidence=0.8)
pose = mp_pose.Pose()
 
# Carrega o vídeo
cap = cv2.VideoCapture(caminho_video)
 
if not cap.isOpened():
    print(f"Erro ao abrir o vídeo no caminho: {caminho_video}")
    exit()
 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(" Fim do vídeo ou erro na leitura dos frames.")
        break
    frame = cv2.resize(frame, (750, 500))
    # Converte para RGB (MediaPipe usa RGB)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    # Processa detecções
    hand_results = hands.process(image)
    face_results = face_detection.process(image)
    pose_results = pose.process(image)
 
    # Volta para BGR para exibir com OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    # Desenha as mãos detectadas
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
 
    # Desenha os rostos detectados
    if face_results.detections:
        for detection in face_results.detections:
            mp_drawing.draw_detection(image, detection)
 
    # Desenha as poses (corpos detectados)
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
 
    # Mostra o frame com as detecções
    cv2.imshow('Detecção de Vítimas em Escombros', image)
 
    # Pressione ESC (tecla 'esc' -> código 27) para encerrar
    if cv2.waitKey(5) & 0xFF == 27:
        break