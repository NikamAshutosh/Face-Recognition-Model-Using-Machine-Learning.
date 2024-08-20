import face_recognition


def recognize_face(image_path, known_faces_encodings, known_face_names):
    # Load the image and encode faces
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Compare faces
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_faces_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)

    return face_locations, face_names