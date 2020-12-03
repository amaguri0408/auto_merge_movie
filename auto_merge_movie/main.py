import sys
import os
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import moviepy.editor as mp

##### 定数の定義 #####
SAMPLE_RANGE = 30   # 開始から何秒をサンプリングするか
FPS = 60            # 一秒にFPS個の候補
MERGIN = 5          # 音声を±何秒までずらすのを候補とするか

##### コマンドライン引数の取得 #####
try:
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
except IndexError:
    print(
'''
Please specify two arguments.
    First argument : movie file
    Second argument : sound file (wav)
'''
    )
    sys.exit()

##### ディレクトリの作成 #####
result_dir = "result3/"
try:
    os.mkdir(result_dir)
except FileExistsError:     # ディレクトリが既に存在してたら何もしない
    pass

##### 動画ファイル読み込み #####
video_file_name = arg1
video = mp.VideoFileClip(video_file_name)
video.audio.write_audiofile(result_dir + "movie_sound.wav")

##### 音声データからフーリエ変換のエッジを出す関数 #####
def peak_sound_file(file_name, out_file_name, q=[50, 90]):
    '''
    引数 
        file_name(str) : 音声ファイルの名前(ディレクトリ込)
        out_file_name(str) : 出力ファイルの名前(ディレクトリ無し)
    返り値
        edge_array(np.array) : 音が変化した時の時間(s)
    '''
    ### ファイルを読み込んでx秒まででモノラルにする
    rate, data = scipy.io.wavfile.read(file_name)
    data = data / 2**15     # 標準化
    data = data[:rate*SAMPLE_RANGE]     # 指定秒数までのデータだけを切り抜き
    data = (data[:, 0] + data[:, 1]) / 2      # モノラル化

    ### そのまま音声データを表示する
    # 横軸(時間)の配列を作成
    time = np.arange(0, data.shape[0]/rate, 1/rate)
    # plot
    fig = plt.figure()
    plt.plot(time, data)
    plt.xlabel("time")
    plt.ylabel("amplitude")
    fig.savefig(result_dir + out_file_name + "_wave.png")   # 保存

    ### 短時間フーリエ変換
    # フレーム長
    fft_size = 1024
    # フレームシフト長
    hop_length = int(fft_size / 4)
    # 実行
    amplitude = np.abs(librosa.core.stft(data, n_fft=fft_size, hop_length=hop_length))
    # 振幅をデシベル単位に変換
    log_power = librosa.core.amplitude_to_db(amplitude)

    # print(log_power)

    # グラフ
    fig = plt.figure()
    librosa.display.specshow(
        log_power, sr=rate, hop_length=hop_length,
        x_axis="time", y_axis="hz", cmap="magma")
    plt.colorbar(format="%+2.0fdB")
    # グラフ保存
    fig.savefig(result_dir + out_file_name + "_fourier_figure.png")

    ### エッジのグラフ
    # tとt-1のlog_powerを引いて2乗して足す
    div_log_power = np.zeros(log_power.shape[1]-1)
    for i in range(log_power.shape[1]-1):
        div_log_power[i] = sum(abs(log_power[:, i+1] - log_power[:, i]) ** 2)
    # plot
    time = np.arange(0, div_log_power.shape[0]/(rate / hop_length), 1/(rate / hop_length))
    fig = plt.figure()
    plt.plot(time, div_log_power)
    plt.xlabel("time")
    plt.ylabel("diffrence")
    fig.savefig(result_dir + out_file_name + "_edge_figure.png")

    # print(div_log_power)

    ### 時間を割り出す 
    tmp = np.percentile(div_log_power, q=q)
    # print(q)
    # print(tmp)
    border1 = tmp[0]
    border2 = tmp[1]
    edge_array = np.array([])
    edge_value_array = np.array([])
    flag = False
    max_value = 0
    max_time = 0
    for i, value in enumerate(div_log_power):
        if value < border1:
            if flag:
                edge_array = np.append(edge_array, max_time / (rate / hop_length))
                edge_value_array = np.append(edge_value_array, max_value)
                max_value = 0
            flag = False
        elif value > border2:
            flag = True
        if flag:
            if value > max_value:
                max_value = value
                max_time = i

    # print(edge_array)

    ### エッジのグラフとピークの点
    # plot
    fig = plt.figure()
    plt.plot(time, div_log_power)
    tmp = np.zeros(edge_array.shape[0])
    plt.plot(edge_array, edge_value_array, "x")
    # plt.plot(edge_array, tmp, "x")
    plt.xlabel("time")
    plt.ylabel("diffrence")
    # plt.show()
    fig.savefig(result_dir + out_file_name + "_edge_figure_a.png")

    return edge_array

##### 音声からエッジを検出する #####
# 動画の音声ファイル
wav_file = result_dir + "movie_sound.wav"
edge_movie_array = peak_sound_file(wav_file, "movie_sound", [90, 90])
wav_file = arg2
edge_sound_array = peak_sound_file(wav_file, "sound", [50, 95])

##### 動画と音声の最適なタイミングを探す #####
# 二分探索のために値を追加
edge_movie_array = np.insert(edge_movie_array, 0, 0)
edge_movie_array = np.append(edge_movie_array, SAMPLE_RANGE)
ans_dt = 0
min_loss = float("inf")
dt_array = np.array([])
loss_array = np.array([])
for i in range(-FPS * MERGIN, FPS * MERGIN):
    dt = i / FPS
    dt_array = np.append(dt_array, dt)
    loss = 0
    for j in edge_sound_array:
        if j < MERGIN or j > SAMPLE_RANGE - MERGIN: continue
        key = j + dt
        ng = 0
        ok = edge_movie_array.shape[0] - 1
        while abs(ok - ng) > 1:
            mid = (ng + ok) // 2
            if edge_movie_array[mid] > key: ok = mid
            else: ng = mid
        loss += min((edge_movie_array[ng] - key) ** 2, (edge_movie_array[ok] - key) ** 2)
    loss_array = np.append(loss_array, loss)
    if loss < min_loss:
        min_loss = loss
        ans_dt = dt
print(ans_dt, min_loss)
### 損失をグラフで表示
# plot
fig = plt.figure()
plt.plot(dt_array, loss_array)
plt.xlabel("dt")
plt.ylabel("loss")
# plt.show()
fig.savefig(result_dir + "dt-loss.png")

##### 音声ファイルの開始位置を変更して保存 #####
wav_file = arg2
rate, data = scipy.io.wavfile.read(wav_file)
if ans_dt >= 0:
    data = np.insert(data, 0, np.zeros((int(rate * ans_dt), 2)), axis=0)
else:
    data = data[abs(int(rate * ans_dt)):]
scipy.io.wavfile.write(result_dir + "out_sound.wav", rate, data)

##### 映像と音声を結合して保存 #####
input_file = arg1
video = mp.VideoFileClip(input_file)
video = video.set_audio(mp.AudioFileClip(result_dir + "out_sound.wav"))
video.write_videofile(result_dir + "main.mp4")

##### 処理のログを記録 #####
path = result_dir + "log.txt"
try:
    with open(path, mode='x') as f:
        pass
except FileExistsError:
    pass

with open(path, mode = "w") as f:
    f.write("movie_file : {}".format(arg1))
    f.write("sound_file : {}".format(arg2))
    f.write("dt : {}".format(ans_dt))