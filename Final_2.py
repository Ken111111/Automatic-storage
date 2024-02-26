import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective, contours, grab_contours
from pyzbar.pyzbar import decode
from snap7.exceptions import Snap7Exception
import snap7
import time
import threading

plc = snap7.client.Client()

prev_dr_H = None
prev_dr_W = None
prev_qr_data = None
threshold = 0.5  # Giả sử một ngưỡng thay đổi nhỏ

def connect_to_plc():
    try:
        plc.connect("192.168.0.1", 0, 1)
        return True
    except Snap7Exception as e:
        print(f"Lỗi kết nối đến PLC: {e}")
        return False

def plc_connection():
    global plc

    max_retries = 3
    current_retry = 0

    while current_retry < max_retries:
        try:
            plc.connect("192.168.0.1", 0, 1)
            print("Đã kết nối đến PLC.")
            return True
        except Exception as e:
            print(f"Lỗi kết nối đến PLC: {e}")
            current_retry += 1
            if current_retry < max_retries:
                print(f"Thử lại kết nối sau 5 giây... (Lần thử lại thứ {current_retry})")
                time.sleep(5)
    
    print("Đã đạt đến số lần thử lại tối đa. Không thể kết nối đến PLC.")
    return False

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) / 2, (ptA[1] + ptB[1]) / 2)

def get_distance_in_pixels(orig, c):
    # Lấy minRect
    box = cv2.minAreaRect(c)
    # Lấy tọa độ các đỉnh của MinRect
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # Sắp xếp các điểm theo trình tự
    box = perspective.order_points(box)

    # Vẽ contour
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    # Tinh toán 4 trung diểm của các cạnh
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # Tính độ lấy 2 chiều
    dc_W = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dc_H = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    return dc_W, dc_H, tltrX, tltrY, trbrX, trbrY

def read_and_preprocess(frame, canny_low=50, canny_high=100, blur_kernel=9, d_e_kernel=3):
    # Chuyển frame sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Làm mờ ảnh
    gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Áp dụng Canny tìm cạnh
    edged = cv2.Canny(gray, canny_low, canny_high)
    edged = cv2.dilate(edged, (d_e_kernel, d_e_kernel), iterations=1)
    edged = cv2.erode(edged, (d_e_kernel, d_e_kernel), iterations=1)

    # Đọc QR code
    qr_data = read_qr_code(frame)
    return frame, edged, qr_data

def find_object_in_pix(orig, edge, area_threshold=3000):
    global prev_dr_H, prev_dr_W, ref_width, prev_qr_data

    # Tìm các Contour trong ảnh
    cnts = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)

    # Sắp xếp các contour từ trái qua phải
    cnts = contours.sort_contours(cnts)[0]

    # Duyệt các contour
    for c in cnts:
        # Nếu contour quá nhỏ -> bỏ qua
        if cv2.contourArea(c) < area_threshold:
            continue

        # Tính toán 2 chiều bằng Pixel
        dc_W, dc_H, tltrX, tltrY, trbrX, trbrY = get_distance_in_pixels(orig, c)

        # Nếu QR data đã được đọc
        qr_data = prev_qr_data

        if qr_data:
            # Cập nhật giá trị `ref_width` dựa trên giá trị của QR code
            if qr_data == '1':
                ref_width = 14.5
                focal_length = 530
            if qr_data == '2':
                ref_width = 13.0
                focal_length = 345
            if qr_data == '3':
                ref_width = 12.0
                focal_length = 490

            # Tính toán kích thước thật dựa vào kích thước pixel và số P
            P = ref_width / dc_H
            dr_W = dc_W * P
            dr_H = dc_H * P
            distance = 37 - (ref_width * focal_length) / dc_W
            # Hiển thị giá trị ra terminal nếu có giá trị mới
            if prev_dr_H is None or abs(prev_dr_H - dr_H) > threshold or \
                    prev_dr_W is None or abs(prev_dr_W - dr_W) > threshold:
                print("dr_H:", dr_H, "cm")
                print("dr_W:", dr_W, "cm")
                print("Distance:", distance, "cm")
            # Cập nhật giá trị trước đó
            prev_dr_H = dr_H
            prev_dr_W = dr_W
            send_qr_data_to_plc(qr_data)
            # Ve kich thuoc len hinh
            cv2.putText(orig, "{:.1f} cm".format(dr_H), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            cv2.putText(orig, "{:.1f} cm".format(dr_W), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            # Vẽ khoảng cách lên trung tâm hình
            center_x, center_y = int((tltrX + trbrX) / 2), int((tltrY + trbrY) / 2)
            cv2.putText(orig, "{:.1f} cm".format(distance), (center_x - 50, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return orig

def read_qr_code(frame):
    # Giải mã QR code
    decoded_objects = decode(frame)

    # Lấy dữ liệu từ QR code nếu có
    qr_data = None
    if decoded_objects:
        qr_data = decoded_objects[0].data.decode('utf-8')
        print("QR Code Data:", qr_data)

    return qr_data

def send_qr_data_to_plc(qr_data):
    global plc

    max_retries = 3
    current_retry = 0

    while current_retry < max_retries:
        try:
            if not plc.get_connected():
                if not connect_to_plc():
                    # Increment the retry counter and try again
                    current_retry += 1
                    print(f"Thử lại kết nối sau 5 giây... (Lần thử lại thứ {current_retry})")
                    time.sleep(5)
                    continue

            if qr_data is not None:
                int_value = int(qr_data)
                byte_value = int_value.to_bytes(2, 'big')
                plc.db_write(2, 120, byte_value)
                print(f"Đã gửi dữ liệu QR Code '{qr_data}' lên PLC dưới dạng INT.")
            else:
                print("Dữ liệu QR Code không tồn tại.")
            
            # Successfully sent data, break out of the retry loop
            break

        except Snap7Exception as e:
            print(f"Lỗi khi gửi dữ liệu QR Code lên PLC: {e}")

            # Increment the retry counter and try again
            current_retry += 1
            print(f"Thử lại kết nối sau 5 giây... (Lần thử lại thứ {current_retry})")
            time.sleep(5)

def display_image(image):
    cv2.imshow("Processed Frame", image)
    cv2.waitKey(1)

def get_camera_feed():
    global prev_qr_data
    # Mở camera với ID 1 (thường là camera ngoại vi)
    cap = cv2.VideoCapture(1)
    # Kết nối đến PLC
    if not connect_to_plc():
        return

    while True:
        try:
            # Đọc frame từ camera
            ret, frame = cap.read()

            # Thực hiện xử lý trên frame (đọc QR code, vẽ đường viền, hiển thị kích thước)
            image, edged, qr_data = read_and_preprocess(frame)
            if qr_data and qr_data != prev_qr_data:
                prev_qr_data = qr_data
                print("New QR Code Detected:", qr_data)
            image = find_object_in_pix(image, edged)

            # Hiển thị frame sau xử lý trực tiếp trên luồng chính
            display_image(image)

            # Gửi dữ liệu QR code lên PLC
            send_qr_data_to_plc(qr_data)

            # Thoát khỏi vòng lặp khi nhấn phím 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Snap7Exception as e:
            print(f"Lỗi khi thực hiện xử lý hoặc gửi dữ liệu QR Code lên PLC: {e}")

    # Giải phóng tài nguyên và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()

# Bắt đầu chương trình bằng cách gọi hàm get_camera_feed()
get_camera_feed()
