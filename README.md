# TransportBox_PP0
PPOを用いて物体搬送の制御機を構築

Config.py:学習環境，モデルの各パラメータを記載
model.py；ニューラルネットを定義．
env.py：pybulletを用いて3次元物理シミュレータの自作環境を作成
main.py；学習を行う部分．PPOと呼ばれる手法を使用
