import cv2
video_path = 'F:\\XLDownload\\dataSet\\00000.avi'
if __name__ == '__main__':
    cap = cv2.VideoCapture(video_path)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))  # 帧率
    print('帧率：{0}'.format(fps))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 分辨率-宽度
    print('帧宽度：{0}'.format(width))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 分辨率-高度
    print('帧高度：{0}'.format(height))
    frame_counter = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    print('帧数：{0}'.format(frame_counter))
    cap.release()
    cv2.destroyAllWindows()
    duration = frame_counter / fps  # 时长，单位s