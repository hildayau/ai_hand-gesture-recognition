# AI Hand Gesture Recognition

## 1. 開発環境
- Python 3.10
- PyTorch
- OpenCV
- MediaPipe
- cvzone
- NumPy
- Web カメラ

## 2. 環境準備
プロジェクトのフォルダでターミナルを開き、仮想環境を作成します。
```powershell
py -3.10 -m venv .venv310
```

仮想環境を有効化します。
```powershell
.\.venv310\Scripts\Activate.ps1
```

必要なライブラリをインストールします。
```powershell
pip install torch numpy opencv-python mediapipe cvzone
```

※ すでに環境を構築している場合は、仮想環境の作成とインストールは不要です。

## 3. プログラムの実行方法
### ① データ収集

```powershell
python src/collect_data.py
```
Web カメラで手のジェスチャーを撮影し、学習用データを収集します。
- `S`：データを保存
- `Q`：終了

※ 収集するジェスチャーに合わせて、プログラム内のラベルを設定してください。

### ② AI モデルの学習

```powershell
python src/train_model.py
```
収集したデータを使用して AI モデルを学習させます。
学習が完了すると、モデルファイル（`.pth`）が保存されます。

### ③ ジェスチャー認識

```powershell
python src/inference.py
```
Web カメラが起動し、リアルタイムでジェスチャーを認識します。
- Fist（グー）
- Thumbs Up（親指を立てる）
- Peace（ピース）
`Q` キーで終了します。

## 4. 補足
すでに学習済みモデルがある場合は、データ収集とモデル学習を行う必要はありません。
以下のコマンドだけで実行できます。

```powershell
python src/inference.py
```

※ データ収集・学習時のファイルパスと、推論時のモデルパスが一致していることを確認してください。