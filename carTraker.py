import cv2

#Notre image
img_file = 'Car Image.jpeg'
video = cv2.VideoCapture('Tesla Dashcam Accident.mp4')

#Notre Classifieur de Voiture préentrainé
classifier_file = 'car_detector.xml'

#création de la classification de voitures
car_tracker = cv2.CascadeClassifier(classifier_file)

#faire tourner indéfiniment jusqu'a ce que la voiture stop ou se crash ou jsp ^^
while True:
    #lire chaque image de la videa 1 a 1
    (read_successful, frame) = video.read()

    if read_successful:
        #change la couleur du frame en noir et blanc
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # détection de voitures
    cars = car_tracker.detectMultiScale(grayscaled_frame)

    #print(cars)

    # Dessin des rectangles sur les voitures détécté
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

    # affichage de la frame
    cv2.imshow('Car detector', frame)

    # ne se ferme pas automatiquement (va attendre ici dan le code qu'une key soit préssé)
    cv2.waitKey(1) #1milisecond


'''
#creation d'une image opencv
img = cv2.imread(img_file)

#convertion de l'image en noir et blanc (nécessaire pour le "haar Cascade")
black_n_white_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#création de la classification de voitures
car_tracker = cv2.CascadeClassifier(classifier_file)

#détection de voitures
cars = car_tracker.detectMultiScale(black_n_white_image)

#print(cars)

#Dessin des rectangles sur les voitures détécté
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)


#affichage de l'image
cv2.imshow('Car detector', img)

#ne se ferme pas automatiquement (va attendre ici dan le code qu'une key soit préssé)
cv2.waitKey()
'''

print('code completed')
