import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
imgCanvas = np.zeros((480,640,3), np.uint8) #Canvas para pintar
paintMode = True #Modo pintar

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    
    # Flip the image horizontally for a selfie-view display.
    image = cv2.flip(image, 1)
    
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

      #----------------
      # Detect fingers
      #----------------
      
      # list of finger tips locators, 4 is thumb, 20 is pinky finger
      tipIds = [4, 8, 12, 16, 20]
      fingersUp = [0, 0, 0, 0, 0]
      
      lm = hand_landmarks.landmark
      
      # x,y coordinates of pinky tip. Coordinates are normalized to [0.0,1.0] with width and height of the image
      lm[tipIds[4]].x
      lm[tipIds[4]].y

      #height, width and depth (RGB=3) of image
      (h,w,d) = image.shape

      # OpenCV function to draw a circle:
      # cv2.circle(image, center_coordinates, radius in pixels, color (Blue 0-255, Green 0-255, Red 0-255), thickness in pixels (-1 solid))
      # Example: draw a red solid circle of 10 pixel radius in the tip of pinky finger:
      # cv2.circle(image, (int(lm[tipIds[4]].x*w),int(lm[tipIds[4]].y*h)), 10, (0,0,255), -1)

      # OpenCV function to draw text on image
      # cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
      # Example: draw a blue "hello" on the upper left corner of the image
      # cv2.putText(image, "hello", (20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0), thickness = 5)

      # See other OpenCV functions to draw a line or a rectangle:
      #cv2.line(image, (int(lm[tipIds[0]].x*w), int(lm[tipIds[0]].y*h)), (int(lm[tipIds[2]].x*w),int(lm[tipIds[2]].y*h)), (0,0,255), 1) 
      #cv2.line(image, (int(lm[tipIds[0]].x*w), int(lm[tipIds[0]].y*h)), (int(lm[tipIds[3]].x*w),int(lm[tipIds[3]].y*h)), (0,0,255), 1) 
      # cv2.rectangle(image, start_point (top-left), end_point (bottom-right), color, thickness)
      #cv2.putText(image, str(sum(fingersUp)), (20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0), thickness = 5)
      
      #Contador de dedos levantados
      for i in range(4):
        if lm[tipIds[i+1]].y - lm[tipIds[i+1]-1].y < 0:
          fingersUp[i+1] = 1
        else: 
          fingersUp[i+1] = 0

      #Flags
      thumbRight = False #El pulgar está en la derecha
      konShow = False #Kon está mostrado

      #Pulgar a la derecha
      if lm[tipIds[0]].x > lm[tipIds[4]].x:
        thumbRight = True

      #Pulgar arriba
      if lm[tipIds[0]].x - lm[tipIds[0]-1].x < 0 and not thumbRight:
        fingersUp[0] = 1
      elif lm[tipIds[0]].x - lm[tipIds[0]-1].x > 0 and thumbRight:
        fingersUp[0] = 1
      else: 
        fingersUp[0] = 0

      #Mostrar Kon
      if fingersUp == [1,1,0,0,1] and not konShow:
        konShow = True
        #cv2.putText(image, "kon", (30,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0), thickness = 5)
        #kon = cv2.imread('Zorro_anime.png')
        #cv2.imshow('kon', kon)        
        #imgCanvas.fill(0)
      else:
        konShow = False
        kon = cv2.imread('Zorro_anime.png')
        cv2.imshow('kon', kon)
        cv2.destroyWindow('kon')
      #Modo pintar
      if fingersUp == [1,1,0,0,0] and not paintMode:
        paintMode = True
      elif fingersUp == [0,1,1,0,0] and paintMode:
        paintMode = False
      if fingersUp == [0,1,0,0,0] and paintMode:
        cv2.circle(image, (int(lm[tipIds[1]].x*w),int(lm[tipIds[1]].y*h)), 10, (0,0,255), -1)
        cv2.circle(imgCanvas, (int(lm[tipIds[1]].x*w),int(lm[tipIds[1]].y*h)), 10, (0,0,255), -1)
      #Suma de dedos levantados
      if paintMode:
        cv2.putText(image, "Paint mode", (20,450),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), thickness = 2)
      else:
        cv2.putText(image, "Paint mode", (20,450),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,225), thickness = 2)
      cv2.putText(image, str(sum(fingersUp)), (20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0), thickness = 5)      

    #Configuración del canvas
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    image = cv2.bitwise_and(image, imgInv)
    image = cv2.bitwise_or(image, imgCanvas)
    
    #Mostrar imágenes   
    cv2.imshow('Canvas', imgCanvas) 
    cv2.imshow('MediaPipe Hands', image)  
    
    if cv2.waitKey(5) & 0xFF == 27:
      break
      
cap.release()