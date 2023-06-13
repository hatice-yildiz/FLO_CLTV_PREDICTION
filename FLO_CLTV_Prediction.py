#############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
#############################################################

# 1.Business Problem
# 2.Data Preparation
# 3.CLTV Veri Yapısının Oluşturulması
# 4.BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
# 5.CLTV Değerine Göre Segmentlerin Oluşturulması

#############################################################
# 1.Business Problem
#############################################################

#FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
#Şirketin orta uzun vadeli plan yapabilmesi için var
#olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.

# Değişkenler
#
# master_id:Eşsiz müşteri numarası
# order_channel:Alışveriş yapılan platforma ait hangi kanalın kullanıldığı(Android, ios, Desktop, Mobile)
# last_order_channel:En son alışverişin yapıldığı kanal
# first_order_date:Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date:Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online:Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline:Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online:Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline:Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline:Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online:Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12:Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

#############################################################

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width',None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

#############################################################
# 2.Data Preparation
#############################################################

# Adım1:flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.

df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()

#Adım2: Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve
#replace_with_thresholds fonksiyonlarını tanımlayınız.
#Not:cltv hesaplanırken frequency değerleri integer
#olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)

#Adım3: "order_num_total_ever_online", "order_num_total_ever_offline",
#"customer_value_total_ever_offline", "customer_value_total_ever_online"
#değişkenlerinin aykırı değerleri varsa baskılayanız.

df.columns
outlier_check = [col for col in df.columns if "_total_ever_" in col]
outlier_check


print([replace_with_thresholds(df,col) for col in outlier_check])
df.head()
#[replace_with_thresholds(df[col],[col]) for col in df.columns if "_total_ever_" in col]

#replace_with_thresholds(df, "order_num_total_ever_online")
#replace_with_thresholds(df, "order_num_total_ever_offline")
#replace_with_thresholds(df, "customer_value_total_ever_offline")
#replace_with_thresholds(df, "customer_value_total_ever_online")

#Adım4: Omnichannel müşterilerin hem online'dan hem de offline platformlardan
#alışveriş yaptığını ifade etmektedir.Her bir müşterinin toplam alışveriş sayısı
#ve harcaması için yeni değişkenler oluşturunuz.

df["order_num_total_ever"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total_ever"] = df["customer_value_total_ever_online"] + \
                                  df["customer_value_total_ever_offline"]

# Adım4:Değişken tiplerini inceleyiniz.Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df.dtypes
date_columns = [col for col in df.columns if "date" in col]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

#2.yol
#date_columns = df.columns{df.columns.str.contains("date")]
#df[date_columns=df[date_columns].apply(pd.to_datetime)
#3.yıl
#df.loc[:,df.columns.str.contains("date")]=df.loc[:,df.columns.str.contains("date")].apply(pd.to_datetime)


#############################################################
# 3. CLTV Veri Yapısının Oluşturulması
#############################################################

#Adım1:Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

df.info()
today_date = df["last_order_date"].max()+dt.timedelta(2)
df.dtypes

#Adım2: customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı
#yeni bir cltv dataframe'i oluşturunuz.Monetary değeri satın alma başına ortalama değer olarak,
#recency ve tenure değerleri ise haftalık cinsten ifade edilecek.
df.columns
df.head()
cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] =((df["last_order_date"] - df["first_order_date"]).astype("timedelta64[D]")/ 7 )
cltv_df["T_weekly"] = ((today_date - df["first_order_date"]).astype("timedelta64[D]")/ 7)
cltv_df["frequency"] = df["order_num_total_ever"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total_ever"] / df["order_num_total_ever"]
cltv_df.head()

"""
df["recency"] = (df["last_order_date"] - df["first_order_date"] )
dff = df.groupby("master_id").agg({ "recency" : lambda recency : recency,
                                    "first_order_date" : lambda first_order_date : (today_date-first_order_date).dt.days,
                                    "order_num_total_ever" : "sum",
                                    "customer_value_total_ever" : "sum"
                                    })
dff.columns = ["recency_cltv_weekly","T_weekly" ,"frequency","monetary_cltv_avg"]
dff ["monetary_cltv_avg"] = dff ["monetary_cltv_avg"] /dff["frequency"]
dff["recency_cltv_weekly"] = dff["recency_cltv_weekly"] / 7
dff["T_weekly"] = dff["T_weekly"] / 7

dff["T_weekly"].head()
"""
#############################################################
# 4.BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
#############################################################

#Adım1:BG/NBD modelini fit ediniz.

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],cltv_df['recency_cltv_weekly'],cltv_df['T_weekly'])
cltv_df.head()

# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

#Adım2:Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip
# exp_average_valueolarak cltv dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                cltv_df['monetary_cltv_avg'])

cltv_df.head()

#Adım3:6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv = ggf.customer_lifetime_value(bgf,cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],cltv_df['monetary_cltv_avg'],
                                   time=6,freq="W",discount_rate=0.01)

cltv_df["cltv"] = cltv

#Cltvdeğeri enyüksek 20 kişiyi gözlemleyiniz.

cltv_df.sort_values("cltv",ascending=False)[:20]

#############################################################
# 5: CLTV Değerine Göre Segmentlerin Oluşturulması
#############################################################

#Adım1:6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

cltv_df.head()

#Adım2:4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.

cltv_df.groupby("cltv_segment").agg({"recency_cltv_weekly" : ["median"],
                                    "T_weekly" : ["median"],
                                    "frequency" :["median"],
                                    "monetary_cltv_avg" :["median"]}).head()

"""         recency_cltv_weekly    T_weekly frequency monetary_cltv_avg
                          median      median    median            median
cltv_segment                                                            
D                     100.571429  118.142857       3.0         89.495000
C                      77.857143   95.428571       4.0        121.990000
B                      72.571429   89.500000       4.0        156.874167
A                      62.428571   79.142857       5.0        209.957500

"""

cltv_df["cltv_segment_2"] = pd.qcut(cltv_df["cltv"], 3, labels=[ "C", "B", "A"])

"""
          recency_cltv_weekly    T_weekly frequency monetary_cltv_avg
                            median      median    median            median
cltv_segment_2                                                            
C                        94.142857  115.000000       3.0         95.490000
B                        74.142857   91.285714       4.0        137.611250
A                        64.714286   81.428571       5.0        198.023333

"""

cltv_df.head()

cltv_df["cltv_segment_3"] = pd.qcut(cltv_df["cltv"], 5, labels=["E","D", "C", "B", "A"])

cltv_df.groupby("cltv_segment_3").agg({"recency_cltv_weekly" : ["median"],
                                    "T_weekly" : ["median"],
                                    "frequency" :["median"],
                                    "monetary_cltv_avg" :["median"]}).head()

cltv_df.head()
"""
               recency_cltv_weekly    T_weekly frequency monetary_cltv_avg
                            median      median    median            median
cltv_segment_3                                                            
E                       107.142857  119.714286       3.0         84.990000
D                        80.428571   97.428571       3.0        112.652143
C                        75.000000   91.714286       4.0        138.487500
B                        69.714286   87.285714       4.0        167.730000
A                        61.142857   78.428571       5.0        222.342500
"""
#### datasetini 4 segmente böldüğümüzde D segmentinde bulunan müşterilerin recency değerinin yani son satın alma üzerinden geçen zaman ortalaması
#diğer segmentlere göre daha yüksek çıkmış olmasına rağmen D segmentinin bıraktığı para miktarı ortalaması diğer segmentlere göre en düşük çıkmıştır.
#B ve C segmentlerine ait müşterilerin ortalama değerleri birbirine yakın olduğu gözlenmiştir. A segmentine ait müşterilerin
# son satın almaları üzerinden geçen zaman süresi ortalaması daha yüksek olmasına rağmen bıraktıkları paranın ortalaması en yüksek çıkmıştır.
#Bu tablo verilerine göre A segmentine ait müşteriler D segmentine ait müşterilere göre daha az gelmesine rağmen
#D segmenti müşterilerine göre daha çok ortalama para kazandırmış oldukları söylenebilir.
#datasetini 3 segmente böldüğümüzde yine datasetini 4 segmente böldüğümüzdeki gibi A segmentine ait müşterilerin D segmentine ait müşterilere
#göre daha az gelmesine rağmen  D segmenti müşterilerine göre daha çok ortalama para kazandırmış oldukları söylenebilir. Bu veri seti için
#müşteriler 3 kategoride ele alınabilir.datasetini 5 segmente böldüğümüzde E ve D segmentine ait müşteriler ile B ve C segmentine ait
#müşterilerin gelme sıklıkları aynı olup bıraktıkları ortalama para miktarlarının birbirine yakın olduğu söylenebilir. 3 durum karşılaştırıldığında
#müşteriler 3 segmente ayrılıp üç kategoride bir satış pazarlaması yapılabilir.

