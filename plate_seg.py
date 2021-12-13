import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import PIL

pytesseract.pytesseract.tesseract_cmd = R'C:\\Program Files\\Tesseract-OCR\\tesseract'



plt.style.use('dark_background')

#이미지 로드
img = cv2.imread('JOG9221.jpg')

height, width, channel = img.shape #이미지 사이즈 저장

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #gray 로 사진 변환
plt.figure(figsize=(12, 10))
plt.imshow(gray, cmap='gray') 


structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

#가우시안 블러 사용 - 노이즈 줄이기
img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=3) #0,1,2,3,4...

#이미지에 쓰레스 홀드 지정해서 0,255(검은색,흰색) 으로 나뉘게끔 - 분류 쉽게할수있게
img_thresh = cv2.adaptiveThreshold(
    img_blurred, 
    maxValue=255.0, 
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    thresholdType=cv2.THRESH_BINARY_INV, 
    blockSize=19, 
    C=9
)

plt.figure(figsize=(12, 10))
plt.imshow(img_thresh, cmap='gray')

#윤곽선 찾기
contours, _= cv2.findContours(img_thresh,mode=cv2.RETR_TREE,
                              method=cv2.CHAIN_APPROX_SIMPLE)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

#윤곽선 그리기 
cv2.drawContours(temp_result, contours=contours, contourIdx=-1, 
                 color=(255, 255, 255))

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

contours_dict = []


for contour in contours:
    x, y, w, h = cv2.boundingRect(contour) #윤곽선을 감싸는 사각형 구하기 - x,y 너비 높이 저장
    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h),  
                  color=(255, 255, 255), thickness=2)
    
    #빈리스트에 정보들 저장
    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2), #중심좌표
        'cy': y + (h / 2)
    })

plt.figure(figsize=(12, 10))
plt.imshow(temp_result, cmap='gray')

##
#윤곽선 사각형의 최소 넓이
MIN_AREA = 80
#최소 너비와 높이
MIN_WIDTH, MIN_HEIGHT = 2, 8
#비율의 min,max
MIN_RATIO, MAX_RATIO = 0.25, 1.0

#위 조건에 부합하는 애들 리스트에 저장
possible_contours = []

cnt = 0
for d in contours_dict:
    area = d['w'] * d['h'] #넓이
    ratio = d['w'] / d['h'] #비율
    
    if area > MIN_AREA \
    and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
    and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt += 1
        possible_contours.append(d) #조건에 맞는 애들만 인덱스 포함하여 저장
        
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for d in possible_contours:

    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), 
                  color=(255, 255, 255), thickness=2)

#결국 번호판의 글자처럼 생긴 애들만 남는것 확인가능
plt.figure(figsize=(12, 10))
plt.imshow(temp_result, cmap='gray')


##사각형들의 배열을 보고  번호판일 가능성이 높은 애들만 뽑기

MAX_DIAG_MULTIPLYER = 5  #첫번째 사각형과 두번째 사각형의 중심 차이가 첫번째 사각형의 대각선의 5배 안쪽에 있어야함
MAX_ANGLE_DIFF = 12.0    #두 사각형의 중심을 그었을때 쎄타(각도) 의 최댓값
MAX_AREA_DIFF = 0.5      #면적차이 (0.5보다 크면 인정x)
MAX_WIDTH_DIFF = 0.8     #너비차이 (..)
MAX_HEIGHT_DIFF = 0.2    #높이차이
MIN_N_MATCHED = 3       #위에 조건을 만족하는 애들이 최소 3개 이상

#재귀함수로 계속 찾기위해 함수로 지정
def find_chars(contour_list):
    matched_result_idx = [] #최종 인덱스
    
    for d1 in contour_list:  #사각형1
        matched_contours_idx = []
        for d2 in contour_list:  #사각형2
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx']) #중심점 차이의 x축
            dy = abs(d1['cy'] - d2['cy']) #중심점 차이의 y축

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2) #d1(사각형1)의 대각길이

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']])) #두 중점 차이의 거리
            if dx == 0: #가로차이가 0이면
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx)) #쎄타(각도) 구하기
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h']) #면적의비율
            width_diff = abs(d1['w'] - d2['w']) / d1['w']                                #너비의비율
            height_diff = abs(d1['h'] - d2['h']) / d1['h']                               #높이의비율

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        matched_contours_idx.append(d1['idx']) #번호판의 일부인 d1도 마지막으로 넣기

        if len(matched_contours_idx) < MIN_N_MATCHED: #후보군의 길이가 MIN_N_MATCHED 보다 작으면 제외
            continue

        matched_result_idx.append(matched_contours_idx) #위과정 다통과하면 최종 후보군에 넣기

        unmatched_contour_idx = [] #최종 후보군에 못든 애들도 한번더 비교해보기
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(possible_contours,        #unmatched_contour_idx와 같은 인덱스의 값만 추출
                                    unmatched_contour_idx)
        
        recursive_contour_list = find_chars(unmatched_contour) #재귀함수로 계속 돌려버리기
        
        for idx in recursive_contour_list: #살아남은 애들 다시 넣어주기
            matched_result_idx.append(idx)

        break

    return matched_result_idx
    
result_idx = find_chars(possible_contours)

matched_result = []
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for r in matched_result:
    for d in r:

        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), 
                      pt2=(d['x']+d['w'], d['y']+d['h']), 
                      color=(255, 255, 255), thickness=2)

plt.figure(figsize=(12, 10))
plt.imshow(temp_result, cmap='gray')


#번호판 영역 추출
PLATE_WIDTH_PADDING = 1.3
PLATE_HEIGHT_PADDING = 1.5
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10

plate_imgs = []
plate_infos = []

#affine transform을 사용하여 번호판 똑바로 돌리기
for i, matched_chars in enumerate(matched_result):
    sorted_chars = sorted(matched_chars, key=lambda x: x['cx']) #순서대로 정렬

    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2 #센터x좌표
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2 #센터y좌표

    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

    sum_height = 0
    for d in sorted_chars:
        sum_height += d['h']

    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy'] #삼각형 높이
    triangle_hypotenus = np.linalg.norm(                             #빗변
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )

    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus)) #번호판을 똑바로 돌리기위해 각도 구하기

    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)  #로테이션 메트릭스 구하기
 
    img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height)) #각도 만큼 회전

    #번호판 부분만 자르기
    img_cropped = cv2.getRectSubPix( 
        img_rotated, 
        patchSize=(int(plate_width), int(plate_height)), 
        center=(int(plate_cx), int(plate_cy))
    )

    if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
        continue

    plate_imgs.append(img_cropped)
    plate_infos.append({
        'x': int(plate_cx - plate_width / 2),
        'y': int(plate_cy - plate_height / 2),
        'w': int(plate_width),
        'h': int(plate_height)
    })

    plt.subplot(len(matched_result), 1, i+1)
    plt.imshow(img_cropped, cmap='gray')


#
longest_idx, longest_text = -1, 0        
plate_chars = []

#한번더 쓰레스홀딩 (OTSU 사용) 
for i, plate_img in enumerate(plate_imgs):
    plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
    _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    contours, _= cv2.findContours(plate_img,mode=cv2.RETR_LIST,method=cv2.CHAIN_APPROX_SIMPLE)
    plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
    plate_max_x, plate_max_y = 0, 0

    #한번더 사각형 검출(위에서한것 똑같음)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        area = w * h
        ratio = w / h

        #설정값이랑 비교
        if area > MIN_AREA \
        and w > MIN_WIDTH and h > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
            if x < plate_min_x:
                plate_min_x = x
            if y < plate_min_y:
                plate_min_y = y
            if x + w > plate_max_x:
                plate_max_x = x + w
            if y + h > plate_max_y:
                plate_max_y = y + h
             
    #최종결과물
    img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
    
    #글씨 좀더 잘읽게 가우시안블러로 노이즈 제거
    img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
    #쓰레스홀딩 한번더
    _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #이미지에 여백주기(패딩)
    img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

    #pytesseract로 이미지 글자 읽기 (psm 7: 이미지가 한줄로 있다)
    chars = pytesseract.image_to_string(img_result, lang='eng', config='--psm 7')

    arr = chars.split('\n')[0:-1]
    chars = '\n'.join(arr)

    
    result_chars = ''
    has_digit = False
    for c in chars:
        print(c)
        if ord('A') <= ord(c) <= ord('Z') or c.isdigit(): #영어나 숫자가 포함되어있는지  
            if c.isdigit():
                has_digit = True
            result_chars += c
    
    print(result_chars)
    plate_chars.append(result_chars)

    if has_digit and len(result_chars) > longest_text: #가장 긴부분 가져오기
        longest_idx = i

    plt.subplot(len(plate_imgs), 1, i+1)
    plt.imshow(img_result, cmap='gray')


##
info = plate_infos[longest_idx]
chars = plate_chars[longest_idx]

print(chars)

img_out = img.copy()

cv2.rectangle(img_out, pt1=(info['x'], info['y']), pt2=(info['x']+info['w'], info['y']+info['h']), color=(255,0,0), thickness=2)

#cv2.imwrite(chars + '.jpg', img_out)

plt.figure(figsize=(12, 10))
plt.imshow(img_out)


