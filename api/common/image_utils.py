
import base64
import cv2
import numpy as np

def readb64(encoded_data):
    #encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def imgtob64(image_path):
    with open(image_path, 'rb') as file:
        image_read = file.read() 
        image_64_encode = base64.encodestring(image_read)
    return image_64_encode.decode('ascii')

def img_cv_tob64(image):
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)
    return jpg_as_text


if __name__ == '__main__':
    img = cv2.imread('/home/user/tensorrt_demos/dockers/plate-detection-api/mini_truck.jpg')

    text = img_cv_tob64(img)
    img = readb64(text)
    # jpg_original = base64.b64decode(jpg_as_text)
    # jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    # image_buffer = cv2.imdecode(jpg_as_np, flags=1)
    # q = np.frombuffer(jpg_original, dtype=np.uint8)
    # img = cv2.imdecode(q, cv2.IMREAD_COLOR)

    # encoded = img_cv_tob64(img)
    # a = readb64(encoded)

    cv2.imwrite('/home/user/tensorrt_demos/dockers/plate-detection-api/test.png',img)
