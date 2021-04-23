import cv2

#Notre image
img_file = 'Car Image.jpeg'

#Notre Classifieur de Voiture préentrainé
classifier_file = 'car_detector.xml'

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
cv2.imshow('Car detector', black_n_white_image)

#ne se ferme pas automatiquement (va attendre ici dan le code qu'une key soit préssé)
cv2.waitKey()
print('code completed')
