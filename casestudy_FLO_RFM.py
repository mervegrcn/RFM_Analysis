
###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# GÖREVLER
###############################################################

###############################################################
# GÖREV 1: Veriyi  Hazırlama ve Anlama (Data Understanding)
###############################################################

# 1. flo_data_20K.csv verisini okuyunuz.

import datetime as dt
import pandas as pd

df_ = pd.read_csv('/Users/mervegurcan/PycharmProjects/pythonProject/DATASETS/flo_data_20k.csv')
df = df_.copy()

#tum kolonlar gozuksun
pd.set_option('display.max_columns', None)

#tum rowlar gozuksun
#pd.set_option('display.max_rows', None)

#float tiplerde virgulden sonra kac basamagin gozukmesini belirleme (%.3f)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# 2. Veri setinde
        # a. İlk 10 gözlem,
        # b. Değişken isimleri,
        # c. Boyut,
        # d. Betimsel istatistik,
        # e. Boş değer,
        # f. Değişken tipleri, incelemesi yapınız.

df.head(10)
df.columns
df.shape
df.describe().T
df.info()
df.isnull().sum()
df.dtypes
df["master_id"].nunique()

df.skew

# 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df.head()
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


# 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df.dtypes

for i in df.columns:
    if "date" in i:
        df[i] = df[i].apply(pd.to_datetime)


# df["last_order_date"] = df["last_order_date"].apply(pd.to_datetime)


# 5. Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısı ve toplam harcamaların dağılımına bakınız.

df.head()

df.groupby("order_channel").agg({"order_num_total" : ["count", "mean"],
                              "customer_value_total" : ["count", "mean"]})

# 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

df[["master_id", "customer_value_total"]].sort_values(by="customer_value_total", ascending=False).head(10)

# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.

df[["master_id", "order_num_total"]].sort_values(by="order_num_total", ascending=False).head(10)

# 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.

df1 = df_.copy()

def data_prep(df):

    df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

    for i in df.columns:
        if "date" in i:
            df[i] = df[i].apply(pd.to_datetime)

    print(df.dtypes)

    df.groupby("order_channel").agg({"order_num_total": ["count", "mean"],
                                     "customer_value_total": ["count", "mean"]})


    print(df.sort_values(by="customer_value_total", ascending=False).head(10))
    print(df.sort_values(by="order_num_total", ascending=False).head(10))

    return df

data_prep(df = df1)

###############################################################
# GÖREV 2: RFM Metriklerinin Hesaplanması
###############################################################

# Recency (Musterinin yeniligi): Analizin yapildigi tarih - ilgili musterinin son satin alim yaptigi tarih
# Frequency (siklik): musterinin yaptigi toplam satin alma
# Monetary (parasal): musterinin yaptigi toplam satin almalar sonrasinda biraktigi toplam para

# Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi

df["last_order_date"].max()

analysis_date = dt.datetime(2021, 6, 1)

# customer_id, recency, frequency ve monetary değerlerinin yer aldığı yeni bir rfm dataframe

rfm = df.agg({"master_id": lambda master_id: master_id,
            "last_order_date": lambda last_order_date: (analysis_date - last_order_date).days,
            "order_num_total": lambda order_num_total: order_num_total,
            "customer_value_total": lambda customer_value_total: customer_value_total})

rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

rfm.describe().T

###############################################################
# GÖREV 3: RF ve RFM Skorlarının Hesaplanması (Calculating RF and RFM Scores)
###############################################################

#  Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çevrilmesi ve
# Bu skorları recency_score, frequency_score ve monetary_score olarak kaydedilmesi

# RFM score belirlerken, Frequency ve monetary de buyuk sayilar 5 puani, kucuk sayilar 1 puani alir fakat recencyde durum farkli
#2 boyutlu bir gorselde RFM skoru hesaplamak icin R ve F degerleri yeterlidir

rfm["recency_score"] = pd.qcut(rfm["recency"], q=5, labels = [5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method = "first"), q=5, labels = [1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"].rank(method = "first"), q=5, labels = [1, 2, 3, 4, 5])

#rank methodu mantığı: diyelimki elimizde 10 bin data var ve bu datanın 5'e bölünmesini istedik. Her bölmede 2000 data olucak
# frequency'de 8 gelen 3 bin tane datamız var. Yani bunu bir aralığa koymamız lazım ama aralıklarımız 2 binlik olduğu için
# 3 binin hepsi tek bir aralığa giremicek rank(method = first) ile ilk bölmeye yerleştir kalanı böl demiş oluyoruz
# bu methodu kullanmadığımızda hata alıyoruz çünkü 3 binlik datayı 2binlik bölmeye sığdırmaya çalışıyor.

# recency_score ve frequency_score’u tek bir değişken olarak ifade edilmesi ve RF_SCORE olarak kaydedilmesi

rfm["RF_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))

rfm.describe().T

###############################################################
# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması
###############################################################

# Oluşturulan RFM skorların daha açıklanabilir olması için segment tanımlama ve  tanımlanan seg_map yardımı ile RF_SCORE'u segmentlere çevirme

# r' [1-2][1-2] : birinci elemanda 1 veya 2 gorursen, 2. elemanda 1 veya 2 gorursen hibernating isimlendirmeyi yap
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

###############################################################
# GÖREV 5: Aksiyon zamanı!
###############################################################

# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

rfm.groupby("segment").agg({"recency": "mean",
                            "frequency": "mean",
                             "monetary": "mean"})

#GORSEL ANALIZ

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def visualization(df, col, target):
    sns.barplot(x=df[col], y=df[target], estimator=np.mean)
    plt.show(block=True)

list = ["recency", "frequency", "monetary"]

for i in list:
    visualization(rfm, i, "segment")


sns.scatterplot(x = "recency", y = "frequency", hue = "segment", data = rfm)
plt.show(block=True)


# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulunuz ve müşteri id'lerini csv ye kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Bu müşterilerin sadık  ve
# kadın kategorisinden alışveriş yapan kişiler olması planlandı. Müşterilerin id numaralarını csv dosyasına yeni_marka_hedef_müşteri_id.cvs
# olarak kaydediniz.

df.columns
rfm.columns

#yukarida columns ismini customer_id olarak degistirdigimiz icin master_id merge'lerken hata aliyorduk. o nedenle customer_id'yi master_id olarak degistirdik
rfm.columns = ['master_id', 'recency', 'frequency', 'monetary',
               'recency_score', 'frequency_score', 'monetary_score', 'RF_SCORE', 'segment']

rfm2 = pd.merge(rfm, df[['interested_in_categories_12','master_id']],on='master_id', how='left')

new_df = pd.DataFrame() #bos dataframe olusturur
new_df["customer_id"] = rfm2[(rfm2["segment"] == "loyal_customers") & (rfm2["interested_in_categories_12"].str.contains("KADIN"))].index

new_df.to_csv("yeni_marka_hedef_musteri_id")

# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
# alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.


target_segments_customer_ids = rfm[rfm["segment"].isin(["cant_loose","hibernating","new_customers"])]["master_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids))
              & ((df["interested_in_categories_12"].str.contains("ERKEK"))
                 |(df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
cust_ids.to_csv("indirim_hedef_müşteri_ids.csv", index=False)

###############################################################
# GÖREV 6: Tum sureci fonksiyonlatiriniz
###############################################################

def create_rfm(df, csv=False):

    #veriyi hazırlama
    df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

    for i in df.columns:
        if "date" in i:
            df[i] = df[i].apply(pd.to_datetime)

    print(df.dtypes)

    df.groupby("order_channel").agg({"order_num_total": ["count", "mean"],
                                     "customer_value_total": ["count", "mean"]})

    print(df.sort_values(by="customer_value_total", ascending=False).head(10))
    print(df.sort_values(by="order_num_total", ascending=False).head(10))


   #RFM metrikleri hesaplanması

    analysis_date = dt.datetime(2021, 6, 1)
    rfm = df.agg({"master_id": lambda master_id: master_id,
                  "last_order_date": lambda last_order_date: (analysis_date - last_order_date).days,
                  "order_num_total": lambda order_num_total: order_num_total,
                  "customer_value_total": lambda customer_value_total: customer_value_total})

    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']


   #RFM skorlarının hesaplanması

    rfm["recency_score"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])

    rfm["RF_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))

    #Segmentlerin İsimlendirilmesi
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalist',
        r'5[4-5]': 'champions'
    }

    rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

    if csv:
        rfm.to_csv("rfm.csv")

    return rfm

rfm_new = create_rfm(df, csv=True)