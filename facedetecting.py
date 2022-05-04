import numpy as np
import cv2
import tensorflow as tf
cap = cv2.VideoCapture(0)
cap.set(3, 480)  # ID 3 = width
cap.set(4, 320)  # ID 4 = height
# Labels — The various outcome possibilities
labels = ["don","tg","fung","other"]
# Loading the model weigths we just downloaded
model = tf.keras.models.load_model("module/keras_model.h5", compile = False)
while True:
    success, image = cap.read()
    if success == False:
        break
    # Necessary to avoid conflict between left and right
    image = cv2.flip(image, 1)
    cv2.imshow("Frame", image)
    # The model takes an image of dimensions (224,224) as input so let’s
    # reshape our image to the same.
    img = cv2.resize(image, (224, 224))

    # Convert the image to a numpy array
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)

    # Normalizing input image
    img = img / 255

    # Predict the class
    prediction = model.predict(img)

    # Map the prediction to the labels
    # Rnp.argmax returns the indices of the maximum values along an axis.
    predicted_labels = labels[np.argmax(prediction[0], axis=-1)]
    #    print(predicted_labels)
    print(predicted_labels, np.argmax(prediction[0], axis=-1), prediction[0])

    # Close all windows if one second has passed and ‘q’ is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release open connections
cap.release()
cv2.destroyAllWindows()