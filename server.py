import socket

# クライアントとの通信に使うポート番号（クライアントと共有）
PORT = 3123

# サーバー用ソケットの作成
# AF_INETはIPv4インターネットプロトコルを表す
# SOCK_STREAMは順序性と信頼性のある双方向のバイトストリームによる接続を表す
servsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# ソケットをアドレスにバインドする
# 今回のサンプルは同ホスト内の通信のみを行う
# そのためsocket.gethostname()で自ホスト名に対してバインドを行っている
# ポート番号はクライアントと同じものを使う
servsock.bind((socket.gethostname(), PORT))

# サーバーを有効にして接続を受け付けるようにする
servsock.listen()

# サーバーのループ
while True:
    # クライアントからの接続を受け付ける
    # clisockにはクライアントに接続済みのソケットが入る
    # addrにはアドレス（タプル）が入る
    print('accept...')
    clisock, addr = servsock.accept()

    # クライアントからデータを受信する
    data = clisock.recv(1024)
    print(data)

    # クライアントへデータを送信する
    clisock.send(b'ok')

    # クライアントとの接続を切る
    # 今回はシングルスレッドでクライアントとの接続を処理している
    # HTTPサーバーのようにしたい場合はマルチスレッドにしてクライアントのソケットを別スレッドで処理するようにする
    clisock.close()