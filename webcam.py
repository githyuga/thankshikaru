from datetime import datetime # 時刻関係のライブラリ
import cv2 # OpenCV のインポート

# VideoCaptureのインスタンスを作成(引数でカメラを選択できる)
cap = cv2.VideoCapture(0)

while True:
    # VideoCaptureから1フレーム読み込む
    ret, frame = cap.read() # 戻り値のframeがimg
    # # 現在時刻の文字列を画像に追加
    # date = datetime.now().strftime("%H:%M.%S")
    # edframe = frame
    # cv2.putText(edframe, date, (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0), 3, cv2.LINE_AA)

    # 加工した画像を表示
    cv2.imshow('Edited Frame', frame)

    # キー入力を1ms待って、keyが「q」だったらbreak
    key = cv2.waitKey(1)&0xff
    if key == ord('q'):
        break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()