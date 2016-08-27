# coding: utf-8

import numpy as np
import random
import time
from sklearn.cluster import KMeans
from pandas import DataFrame
from scipy import stats

NUM_OF_SAMPLE = 100
NUM_OF_TRAIL = 100
NUM_OF_DIM = 5
NUM_OF_RUN = 1
# CLUSTERING_THRESHOLD = 90
CLUSTERING_THRESHOLD = 100 - NUM_OF_DIM * 5

# 一回分の実行で得られたデータの保存用
# 実行時間の平均保持
cent_ave_time_list = []
all_ave_time_list = []
# 精度の保持
cent_percent_list = []
all_percent_list = []
fail_percent_list = []

# リストから添字キー付きの辞書型に変換する関数
def list_to_dict(val_list):
  return dict(zip(range(0,len(val_list)), val_list))

runtime_s = time.time()
for rnum in range(0, NUM_OF_RUN):
    print "==============================RUN{0}==============================".format(rnum+1)
    # 時間統計用
    cent_time_list = []
    all_time_list = []

    # 精度計算用カウント変数
    cnt_cent = 0
    cnt_all = 0
    cnt_fail = 0

    for num in range(0, NUM_OF_TRAIL):
        print "==============================TRIAL{0}==============================".format(num+1)
        # 空の配列で初期化
        samples = np.empty((0,NUM_OF_DIM), int)

        # サンプルを生成
        samples = [np.array([random.randint(0,5) for i in range(0, NUM_OF_DIM)]) for j in range(0, NUM_OF_SAMPLE)]

        print "-----ランダムなデータをクラスタリング-----"
        kmeans_model = KMeans(n_clusters=3, init='k-means++', random_state=None).fit(samples)

        labels = kmeans_model.labels_

        # どの属性ベクトルがどのクラスタに属しているかを表すタプルのリストを作成
        belongs_tuple_list = [(label, sample) for label, sample in zip(labels, samples)]

        # add newdata
        newdata = [random.randint(0, 5) for i in range(0, NUM_OF_DIM)]

        print
        print "-----ランダムなデータを新しく用意-----"
        print newdata

        center_dict = list_to_dict(kmeans_model.cluster_centers_)

        #############################################################
        ### (i)重心のベクトル <--比較--> 追加データの特徴ベクトル ###
        #############################################################
        start = time.time()

        # 辞書内包表記を使って{クラスタNo. : そのクラスタの重心ベクトル} から {クラスタNo, : 新しく追加したデータと重心ベクトルの距離} へ変換
        center_dist_dict = {k: np.linalg.norm(newdata - v) for k,v in center_dict.items()}

        # 上で求めた辞書型のうち、最小の「値」を持つようなの「キー」を求める
        # min関数のkeyオプションにラムダ式を指定する事で、
        # 1番目の要素（辞書型の値）を基準に最小の値を求め、最小の値を持つ0番目の要素（辞書型のキー）を返す

        # 全クラスタの重心と追加ベクトルを比較し、最も重心と追加ベクトルとの距離が近かったクラスタを取得
        res_cent = min(center_dist_dict.items(), key=lambda x: x[1])[0]
        print
        print "-----重心比較法で新しいデータが属すクラスタを予測-----"
        print "ラベル:"
        print res_cent
        print "重心:"
        center_centmethod = center_dict[res_cent]
        print center_centmethod

        # 予測されたクラスタに属すサンプルを取得
        cntmethod_datas = [tup[1] for tup in belongs_tuple_list if tup[0] == res_cent]

        elapsed_time = time.time() - start
        print ("重心比較法 実行時間:\n{0}".format(elapsed_time)) + "[sec]"

        # 重心を用いたクラスタ選択方法での実行時間を記録
        cent_time_list.append(elapsed_time)

        ############################################################
        ### (ii)全てのデータ <--比較--> 追加データの特徴ベクトル ###
        ############################################################
        start = time.time()

        # 辞書内包表記を使って {属しているクラスタラベル : 特徴ベクトル} から
        # {属しているクラスタラベル : 新しく追加したデータと特徴ベクトルの距離} へ変換
        all_dist_dict = [(k, np.linalg.norm(newdata - v)) for (k, v) in belongs_tuple_list]

        # 全特徴ベクトルと追加されたベクトルを比較し、最も近い特徴ベクトルが属していたクラスタを取得
        res_all = min(all_dist_dict, key=lambda x: x[1])[0]
        print
        print "-----全走査法で新しいデータが属すクラスタを予測-----"
        print "ラベル:"
        print res_all
        print "重心:"
        center_allmethod = center_dict[res_all]
        print center_allmethod

        # 予測されたクラスタに属すサンプルを取得
        allmethod_datas = [tup[1] for tup in belongs_tuple_list if tup[0] == res_all]

        elapsed_time = time.time() - start
        print ("全走査法 実行時間:\n{0}".format(elapsed_time)) + "[sec]"

        # 全ベクトル走査を用いたクラスタ選択方法での実行時間を記録
        all_time_list.append(elapsed_time)

        # 実験の最初で用意した特徴ベクトルの集合に、新しいベクトルを追加
        # クラスタリングを再度行い、上の(i)と(ii)でどちらの結果が正しかったかを確認
        # 新しいデータを追加
        samples = np.append(samples, np.array([newdata]), axis = 0)

        # k-meansを実行
        print
        print "-----新しいデータを加え再クラスタリング-----"
        kmeans_model = KMeans(n_clusters=3, init='k-means++', random_state=None).fit(samples)
        labels = kmeans_model.labels_
        new_center_dict = list_to_dict(kmeans_model.cluster_centers_)

        new_belongs_tuple_list = [(label, sample) for label, sample in zip(labels, samples)]

        # newdataをindex(1)に含むタプルのindex(0)を取得する
        res_actually = None
        for t in new_belongs_tuple_list:
            # Numpyの「配列同士」の比較は「配列の要素同士」の比較になるので、
            # 全部の要素が一致する場合を検知したいはall()を使う
            if all(t[1] == newdata):
                res_actually = t[0]

        print "実際のクラスタリングで新しい特徴ベクトルが属すクラスタ:"
        print "ラベル:"
        print res_actually
        print "重心:"
        center_actually = new_center_dict[res_actually]
        print center_actually
        actually_datas = [tup[1] for tup in new_belongs_tuple_list if tup[0] == res_actually]

        # 提案で得たクラスタと実際のクラスタリングで得たクラスタのデータで
        # 同一の要素が割合的にどれだけ含まれていたか計算
        cntmethod_content_rate = len([arr for arr in actually_datas if any(all(arr == x) for x in cntmethod_datas)]) / float(len(actually_datas)) * 100.0
        allmethod_content_rate = len([arr for arr in actually_datas if any(all(arr == x) for x in allmethod_datas)]) / float(len(actually_datas)) * 100.0

        # 施行の結果を表示、カウント
        if ((np.linalg.norm(center_centmethod - center_actually) <= np.float64(1)) & (np.linalg.norm(center_allmethod == center_actually) <= np.float64(1))) | ((cntmethod_content_rate >= CLUSTERING_THRESHOLD) & (allmethod_content_rate >= CLUSTERING_THRESHOLD)):
            cnt_cent = cnt_cent + 1
            cnt_all = cnt_all + 1
            print
            print "#####両方成功#####"
            print "実際のクラスタの、予測したクラスタ要素の含有率 {0} %".format(cntmethod_content_rate) # 同じなので表示するcontent_rateはどっちでもいい
            print
        elif (np.linalg.norm(center_centmethod - center_actually) <= np.float64(1)) | (cntmethod_content_rate >= CLUSTERING_THRESHOLD):
            cnt_cent = cnt_cent + 1
            print
            print "#####重心比較のみ成功#####"
            print "実際のクラスタの、予測したクラスタ要素の含有率 {0} %".format(cntmethod_content_rate)
            print
        elif (np.linalg.norm(center_allmethod - center_actually) <= np.float64(1)) | (allmethod_content_rate >= CLUSTERING_THRESHOLD):
            cnt_all = cnt_all + 1
            print
            print "#####全走査のみ成功#####"
            print "実際のクラスタの、予測したクラスタ要素の含有率 {0} %".format(allmethod_content_rate)
            print
        else:
            cnt_fail = cnt_fail + 1
            print
            print "#####両方失敗#####"
            print "実際のクラスタの、予測したクラスタ要素の含有率 {0} %".format(allmethod_content_rate)
            print "新しいデータが既に存在していたデータ集合の傾向に比べ大きく違っていたため、"
            print "再クラスタリングで大きくクラスタの構造が変わったようです"
            print

    print
    print "==実行結果=="
    print "(i)重心比較法の精度"
    # res使い回し
    res = cnt_cent/float(NUM_OF_TRAIL) * 100.0
    print "{0} %".format(res)
    cent_percent_list.append(res)
    print "(ii)全走査法の精度"
    res = cnt_all/float(NUM_OF_TRAIL) * 100.0
    print "{0} %".format(res)
    all_percent_list.append(res)
    print "(iii)両方とも外れた割合"
    res = cnt_fail/float(NUM_OF_TRAIL) * 100.0
    print "{0} %".format(res)
    fail_percent_list.append(res)
    print
    print "==実行時間 (統計) =="
    data = {1: cent_time_list, 2: all_time_list}
    df = DataFrame(data, index = ["trial" + str(i+1)  for i in np.arange(NUM_OF_TRAIL)])
    print df.describe()
    cent_ave_time_list.append(df.ix[:, 1].mean())
    all_ave_time_list.append(df.ix[:, 2].mean())

print
print "=================================================================="
print "                             =最終結果="
print "=================================================================="
print "==クラスタリング施行回数=="
print NUM_OF_TRAIL
print "==クラスタリング施行対象=="
print "ランダムな数値を持つ {0} 次元ベクトル {1} 個の集合".format(NUM_OF_DIM, NUM_OF_SAMPLE)
print "==閾値=="
print "{0} %".format(CLUSTERING_THRESHOLD)
print "実行時間:{0} [sec]".format(time.time() - runtime_s)
print "========================================"
time_data = {1: cent_ave_time_list, 2: all_ave_time_list}
print DataFrame(time_data, index = ["run" + str(i+1) for i in np.arange(NUM_OF_RUN)]).describe()

acc_data = {1: cent_percent_list, 2: all_percent_list, 3: fail_percent_list}
print DataFrame(acc_data, index = ["run" + str(i+1) for i in np.arange(NUM_OF_RUN)]).describe()
