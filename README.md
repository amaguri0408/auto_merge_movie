# auto_merge_movie
楽器の演奏動画と別で録音した音声を合成するpythonのプログラム  
A python program that synthesizes audio recorded separately from musical instrument performance videos

Qiitaに解説記事を載せてます
https://qiita.com/amaguri0408/private/6c36a84ba2625c20580c

# Requirement
以下のライブラリが必要です
* moviepy 1.0.1
* librosa 0.8.0

# Usage

```bash
python main.py [動画ファイル] [音声ファイル]
```

demo
```
python main.py sample_movie.mp4 sample_sound.wav
```

resultディレクトリが作成され結果が格納されます。
result/main.mp4が出力です。

# Note

* 動画ファイル、音声ファイルともに30秒以上
* 動画ファイルと音声ファイルのずれは前後5秒以内に収める  
※上2つの制約はmain.pyの10-13行目を書き換えることで緩和できます
* 音声ファイルはwav形式
