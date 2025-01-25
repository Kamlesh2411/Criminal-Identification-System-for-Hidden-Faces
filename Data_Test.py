
import cv2

# Initialize the video capture object
video = cv2.VideoCapture(0)  # 0 means the default webcam, you can change it to a video file path if needed

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

name_list = ["", "Mahesh", "Himanshu", "Pratiksha", "", "AKASH", "Sohan", "Juned", "Kamlesh"]

trained_model = cv2.face_LBPHFaceRecognizer.create()
trained_model.read("Trainer.yml")

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        label, confidence = trained_model.predict(face_roi)

        if confidence > 50:
            name = name_list[label]
        else:
            name = "Unknown"

        print(f"Name: {name}, Confidence: {confidence:.2f}")

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

