import cv2

#Notre image
img_file = 'Car Image.jpeg'

#Notre Classifieur de Voiture préentrainé
classifier_file = 'car_detector.xml'

#creation d'une image opencv
img = cv2.imread(img_file)

#affichage de l'image
cv2.imshow('Car detector',img)

#ne se ferme pas automatiquement (va attendre ici dan le code qu'une key soit préssé)
cv2.waitKey()
print('hello world')
