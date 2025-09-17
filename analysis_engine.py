# analysis_engine.py
# SORUMLULUĞU: Tüm analitik hesaplamaları ve modellemeyi yapmak.
import streamlit as st
from fpdf import FPDF
from fpdf.fonts import FontFace
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from itertools import permutations
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import io
import shap
import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime
from sklearn.cluster import DBSCAN
import os
import warnings
from scipy.stats import ttest_ind, levene
from prophet import Prophet
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')

# --- RFM, Segmentasyon, Churn ve CLV Fonksiyonları ---

# PDF Raporları için Şablon Sınıfı
class PDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Sınıf oluşturulur oluşturulmaz fontları tanıtalım.
        try:
            self.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
            self.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
            self.add_font('DejaVu', 'I', 'DejaVuSans-Oblique.ttf', uni=True)
            self.set_default_font('DejaVu')
        except RuntimeError as e:
            print(f"UYARI: Font dosyaları yüklenemedi. Raporlar Arial ile oluşturulacak. Hata: {e}")
            self.set_default_font('Arial')

    def set_default_font(self, font_family):
        self.default_font_family = font_family

    def header(self):
        self.set_font(self.default_font_family, 'B', 15)
        self.cell(0, 10, 'Müşteri Analitik Raporu', 0, 1, 'C')
        self.set_font(self.default_font_family, '', 8)
        self.cell(0, 10, datetime.now().strftime("%d-%m-%Y %H:%M:%S"), 0, 0, 'C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.default_font_family, 'I', 8)
        self.cell(0, 10, f'Sayfa {self.page_no()}', 0, 0, 'C')
        
    def chapter_title(self, title):
        self.set_font(self.default_font_family, 'B', 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 8, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, content):
        self.set_font(self.default_font_family, '', 10)
        self.multi_cell(0, 5, content)
        self.ln()
        
    def add_plotly_fig(self, fig, width=190):
        buf = io.BytesIO()
        fig.write_image(buf, format="png", width=width*3, height=(width/1.618)*3, scale=2)
        buf.seek(0)
        self.image(buf, w=width)
        buf.close()


def rfm_skorlarini_hesapla(dataframe):
    analiz_tarihi = dataframe['Tarih'].max() + timedelta(days=1)
    rfm_df = dataframe.groupby('MusteriID').agg({
        'Tarih': lambda tarih: (analiz_tarihi - tarih.max()).days,
        'MusteriID': lambda id: id.count(),
        'ToplamTutar': 'sum',
        'NetKar': 'sum'
    }).rename(columns={'Tarih': 'Recency', 'MusteriID': 'Frequency', 
                       'ToplamTutar': 'Monetary', 'NetKar': 'ToplamNetKar'})
    
    if rfm_df.shape[0] < 5:
        for col in ['R_Score', 'F_Score', 'M_Score', 'MPS']:
            rfm_df[col] = 0
        return rfm_df

    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1])
    rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    
    for col in ['R_Score', 'F_Score', 'M_Score']:
        rfm_df[col] = rfm_df[col].astype(int)
        
    w_r, w_f, w_m = 0.20, 0.40, 0.40
    rfm_df['MPS'] = ((w_r * rfm_df['R_Score'] + w_f * rfm_df['F_Score'] + w_m * rfm_df['M_Score']) / 5) * 100
    return rfm_df.sort_values('MPS', ascending=False)

def musterileri_segmentle(rfm_df):
    skor_araliklari = [0, 40, 60, 80, 90, 101]
    segment_etiketleri = ['Kayıp Müşteriler', 'Riskli Müşteriler', 'Sadık Müşteriler', 'Potansiyel Şampiyonlar', 'Şampiyonlar']
    rfm_df['Segment'] = pd.cut(rfm_df['MPS'], bins=skor_araliklari, labels=segment_etiketleri, right=False)
    return rfm_df

def churn_tahmin_modeli_olustur(rfm_df, churn_limiti_gun=180):
    rfm_df['Churn'] = rfm_df['Recency'].apply(lambda x: 1 if x > churn_limiti_gun else 0)
    if len(rfm_df['Churn'].unique()) < 2:
        rfm_df['Churn_Olasiligi'] = 0
        return rfm_df, 0
    X = rfm_df[['Recency', 'Frequency', 'Monetary']]
    y = rfm_df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    dogruluk = accuracy_score(y_test, model.predict(X_test))
    rfm_df['Churn_Olasiligi'] = model.predict_proba(X)[:, 1]
    return rfm_df.sort_values('Churn_Olasiligi', ascending=False), model, dogruluk

def clv_hesapla(rfm_df): # kar_marji parametresi kaldırıldı
    """
    Gerçek net kar verisine dayanarak CLV hesaplar.
    """
    toplam_churn_orani = rfm_df['Churn'].mean()
    if toplam_churn_orani == 0: toplam_churn_orani = 0.001
        
    # Ortalama Sipariş Tutarı hala ciro bazlı olabilir, bu kalsın.
    rfm_df['Ortalama_Siparis_Tutari'] = rfm_df['Monetary'] / rfm_df['Frequency']
    
    musteri_omru = 1 / toplam_churn_orani
    
    # YENİ ve DOĞRU CLV HESAPLAMASI
    # Artık varsayımsal ciro ve kar marjı yerine, müşterinin geçmişteki
    # gerçek toplam net karını geleceğe yansıtıyoruz.
    rfm_df['CLV_Net_Kar'] = rfm_df['ToplamNetKar'] * musteri_omru
    
    return rfm_df.sort_values('CLV_Net_Kar', ascending=False)

def market_basket_analizi_yap(df, min_support=0.05, max_urun_sayisi=300):
    """
    Pazar sepeti (market basket) analizi.
    Daha az bellek kullanan, büyük veri setlerine uygun optimize edilmiş versiyon.
    """

    # En çok satan ürünlerden sadece ilk N tanesini al (varsayılan 300)
    top_urunler = df['UrunKodu'].value_counts().nlargest(max_urun_sayisi).index
    df = df[df['UrunKodu'].isin(top_urunler)]

    # Sepet -> Müşteri x Ürün matrisi
    sepet_df = pd.crosstab(df['MusteriID'], df['UrunKodu'])
    sepet_df = (sepet_df > 0).astype('uint8')   # bool yerine küçük int tip → RAM tasarrufu

    if sepet_df.shape[0] < 10:
        return pd.DataFrame()

    # Apriori algoritmasını low_memory ile çalıştır
    sik_kullanilan_urunler = apriori(sepet_df, 
                                     min_support=min_support, 
                                     use_colnames=True, 
                                     low_memory=True)

    if sik_kullanilan_urunler.empty:
        return pd.DataFrame()

    # Kuralları hesapla ve filtrele
    kurallar = association_rules(sik_kullanilan_urunler, metric="lift", min_threshold=1)
    kurallar = kurallar[
        (kurallar['lift'] >= 1.1) & (kurallar['confidence'] >= 0.1)
    ].sort_values('lift', ascending=False)

    return kurallar

def kohort_analizi_yap(df, metric='retention', period='M'):
    """
    NİHAİ VERSİYON: Farklı metrikleri (retention, avg_spend) ve farklı periyotları (aylık, çeyreklik)
    hesaplayabilen kohort analizi fonksiyonu.
    """
    df_c = df.copy()
    # Periyot parametresini kullanarak gruplama yap
    df_c['SiparisDonemi'] = df_c['Tarih'].dt.to_period(period)
    df_c['Kohort'] = df_c.groupby('MusteriID')['Tarih'].transform('min').dt.to_period(period)
    
    # İki periyot arasındaki farkı daha sağlam bir yöntemle hesapla
    df_c['Donem_Indeksi'] = df_c.apply(lambda row: (row['SiparisDonemi'] - row['Kohort']).n + 1, axis=1)
    
    grup = df_c.groupby(['Kohort', 'Donem_Indeksi'])
    
    if metric == 'retention':
        kohort_data = grup['MusteriID'].nunique().reset_index()
        kohort_sayilari = kohort_data.pivot_table(index='Kohort', columns='Donem_Indeksi', values='MusteriID')
        kohort_buyuklugu = kohort_sayilari.iloc[:, 0]
        heatmap_matrix = kohort_sayilari.divide(kohort_buyuklugu, axis=0)
    
    elif metric == 'avg_spend':
        kohort_data = grup.agg(
            ToplamHarcama=('ToplamTutar', 'sum'),
            AktifMusteri=('MusteriID', 'nunique')
        ).reset_index()
        kohort_data['OrtalamaHarcama'] = kohort_data.apply(
            lambda x: x['ToplamHarcama'] / x['AktifMusteri'] if x['AktifMusteri'] > 0 else 0, axis=1
        )
        heatmap_matrix = kohort_data.pivot_table(index='Kohort', columns='Donem_Indeksi', values='OrtalamaHarcama')
        
    else: # Varsayılan olarak retention
        kohort_data = grup['MusteriID'].nunique().reset_index()
        kohort_sayilari = kohort_data.pivot_table(index='Kohort', columns='Donem_Indeksi', values='MusteriID')
        kohort_buyuklugu = kohort_sayilari.iloc[:, 0]
        heatmap_matrix = kohort_sayilari.divide(kohort_buyuklugu, axis=0)

    # Farklı periyot formatlarını desteklemek için .astype(str) kullan
    heatmap_matrix.index = heatmap_matrix.index.astype(str)
    return heatmap_matrix

def satis_tahmini_yap(zaman_serisi_df, ay_sayisi=6):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(zaman_serisi_df)
    future = model.make_future_dataframe(periods=ay_sayisi, freq='ME')
    forecast = model.predict(future)
    forecast['yhat'] = np.maximum(0, forecast['yhat'])
    forecast['yhat_lower'] = np.maximum(0, forecast['yhat_lower'])
    forecast['yhat_upper'] = np.maximum(0, forecast['yhat_upper'])
    return model, forecast

def tahmin_grafigini_ciz(model, forecast, musteri_id=None, return_fig=False):
    fig = model.plot(forecast, xlabel="Tarih", ylabel="Satış Tutarı")
    ax = fig.gca()
    ax.set_title(f"{musteri_id if musteri_id else ''} Satış Tahmini ve Trendler")
    if return_fig: return fig
    else: plt.show()

def prophet_tahmin(df, tahmin_periyodu=6, gelecek_regresorler=None):
    """
    GÜNCELLENMİŞ: Prophet modeli ile tahmin yapar. Artık geleceğe yönelik harici regresör 
    varsayımlarını da kabul edebilir.
    """
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.05)
    
    # Modelin kullanabileceği regresörleri (varsa) ekle
    regresor_sutunlari = [col for col in ['musteri_sayisi', 'toplam_miktar'] if col in df.columns]
    for sutun in regresor_sutunlari:
        model.add_regressor(sutun)
    
    model.fit(df)
    future = model.make_future_dataframe(periods=tahmin_periyodu, freq='ME')
    
    # Gelecek için regresör değerlerini doldur
    if regresor_sutunlari:
        # Önce geçmiş verileri birleştir
        future = pd.merge(future, df[['ds'] + regresor_sutunlari], on='ds', how='left')
        
        # Gelecek varsayımlarını (varsa) kullan
        if gelecek_regresorler:
            for sutun, deger in gelecek_regresorler.items():
                if sutun in future.columns:
                    future.loc[future['ds'] > df['ds'].max(), sutun] = deger
        
        # Kalan boşlukları (geçmişin ortalamasıyla) doldur
        for sutun in regresor_sutunlari:
            future[sutun].fillna(df[sutun].mean(), inplace=True)

    forecast = model.predict(future)
    forecast['yhat'] = np.maximum(0, forecast['yhat'])
    forecast['yhat_lower'] = np.maximum(0, forecast['yhat_lower'])
    forecast['yhat_upper'] = np.maximum(0, forecast['yhat_upper'])
    
    return model, forecast

def arima_tahmin(df, tahmin_periyodu=6):
    """ARIMA modeli ile tahmin yapar ve negatif sonuçları sıfırlar."""
    best_aic, best_params = np.inf, None
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    res = ARIMA(df['y'], order=(p,d,q)).fit()
                    if res.aic < best_aic: best_aic, best_params = res.aic, (p,d,q)
                except: continue
    
    model = ARIMA(df['y'], order=best_params if best_params else (1,1,1)).fit()
    forecast = model.get_forecast(steps=tahmin_periyodu)
    conf = forecast.conf_int()
    forecast_df = pd.DataFrame({'tahmin': forecast.predicted_mean, 'alt_sinir': conf.iloc[:, 0], 'ust_sinir': conf.iloc[:, 1]})
    forecast_df['tahmin'] = np.maximum(0, forecast_df['tahmin'])
    forecast_df['alt_sinir'] = np.maximum(0, forecast_df['alt_sinir'])
    forecast_df['ust_sinir'] = np.maximum(0, forecast_df['ust_sinir'])
    
    return model, forecast_df, best_params

def sarima_tahmin(df, tahmin_periyodu=6):
    """SARIMA modeli ile mevsimsel tahmin yapar ve negatif sonuçları sıfırlar."""
    try:
        model = SARIMAX(df['y'], order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
        forecast = model.get_forecast(steps=tahmin_periyodu)
        conf = forecast.conf_int()
        forecast_df = pd.DataFrame({'tahmin': forecast.predicted_mean, 'alt_sinir': conf.iloc[:, 0], 'ust_sinir': conf.iloc[:, 1]})
        forecast_df['tahmin'] = np.maximum(0, forecast_df['tahmin'])
        forecast_df['alt_sinir'] = np.maximum(0, forecast_df['alt_sinir'])
        forecast_df['ust_sinir'] = np.maximum(0, forecast_df['ust_sinir'])
        
        return model, forecast_df
    except:
        return None, None

def ensemble_tahmin(df, tahmin_periyodu=6):
    tahminler, modeller = [], {}
    _, prophet_f = prophet_tahmin(df.copy(), tahmin_periyodu)
    tahminler.append(prophet_f['yhat'].tail(tahmin_periyodu).values); modeller['Prophet'] = prophet_f['yhat'].tail(tahmin_periyodu).values
    
    _, arima_f, _ = arima_tahmin(df.copy(), tahmin_periyodu)
    tahminler.append(arima_f['tahmin'].values); modeller['ARIMA'] = arima_f['tahmin'].values
    
    _, sarima_f = sarima_tahmin(df.copy(), tahmin_periyodu)
    if sarima_f is not None:
        tahminler.append(sarima_f['tahmin'].values); modeller['SARIMA'] = sarima_f['tahmin'].values
    
    modeller['Ensemble'] = np.mean(tahminler, axis=0)
    return modeller

def what_if_analizi(df, senaryo_tipi, degisim_orani):
    df_kopya = df.copy()
    if senaryo_tipi == "Fiyat Değişimi":
        df_kopya['y'] *= (1 + degisim_orani / 100)
    elif senaryo_tipi == "Müşteri Sayısı Değişimi" and 'musteri_sayisi' in df_kopya.columns:
        df_kopya['musteri_sayisi'] *= (1 + degisim_orani / 100)
    elif senaryo_tipi == "Kampanya Etkisi":
        df_kopya.loc[df_kopya.index[-3:], 'y'] *= 1.2
    return df_kopya

def musteri_yolculugu_analizi_yap(temiz_df, sonuclar_df, periyot='Q'):
    """
    NİHAİ VERSİYON: Müşteri segmentlerinin zamana bağlı değişimini analiz eder.
    Bu versiyon, YENİ, MEVCUT ve KAYIP (CHURN) müşteri akışlarını da hesaplar.
    """
    df_yolculuk = temiz_df.copy()
    
    zaman_araliklari = pd.to_datetime(df_yolculuk['Tarih'].dt.to_period(periyot).sort_values().unique().to_timestamp())
    
    tum_segmentler = []
    
    for donem_sonu in zaman_araliklari:
        donem_verisi = df_yolculuk[df_yolculuk['Tarih'] <= donem_sonu]
        if not donem_verisi.empty:
            rfm_df = rfm_skorlarini_hesapla(donem_verisi)
            if rfm_df.empty or 'MPS' not in rfm_df.columns:
                continue
            segmentli_df = musterileri_segmentle(rfm_df)
            segmentli_df['Donem'] = donem_sonu.to_period(periyot)
            tum_segmentler.append(segmentli_df[['Segment', 'Donem']].reset_index())

    if not tum_segmentler:
        return pd.DataFrame(), pd.DataFrame(), None, None

    yolculuk_df = pd.concat(tum_segmentler)
    
    # CLV verisini yolculuk verisiyle birleştir
    yolculuk_df_clv = pd.merge(yolculuk_df, sonuclar_df[['CLV_Net_Kar']], on='MusteriID', how='left')
    
    # Pivot'u oluştururken sadece segment bilgilerini alalım
    yolculuk_pivot = yolculuk_df.pivot_table(index='MusteriID', columns='Donem', values='Segment', aggfunc='first')
    
    # CLV'yi pivot'a en son ekleyelim
    yolculuk_pivot = yolculuk_pivot.merge(sonuclar_df[['CLV_Net_Kar']], left_index=True, right_index=True)
    
    # Bu fonksiyon artık sadece pivot'u döndürecek. Geçiş hesaplaması arayüzde yapılacak.
    # Bu, yeni/kayıp müşteri mantığını daha temiz bir şekilde yönetmemizi sağlar.
    return yolculuk_pivot, pd.DataFrame(), None, None # Geriye dönük uyumluluk için boş df döndürüyoruz

def urun_tavsiyesi_uret(birliktelik_kurallari, musteri_urunleri_listesi):
    """
    Bir müşterinin satın aldığı ürünlere ve genel birliktelik kurallarına dayanarak
    kişiselleştirilmiş ürün tavsiyeleri üretir.
    """
    musteri_sepeti = set(musteri_urunleri_listesi)
    potansiyel_tavsiyeler = []

    if birliktelik_kurallari.empty:
        return pd.DataFrame()

    for index, rule in birliktelik_kurallari.iterrows():
        antecedents = set(rule['antecedents'])
        consequents = set(rule['consequents'])

        if antecedents.issubset(musteri_sepeti) and not consequents.issubset(musteri_sepeti):
            # öğeleri stringe çevir (herhangi bir iterable olsa da düzgün metin döner)
            tavsiye_edilen_str = ', '.join(map(str, consequents))
            sebep_str = ', '.join(map(str, antecedents))

            potansiyel_tavsiyeler.append({
                'Tavsiye Edilen Ürün': tavsiye_edilen_str,
                'Satın Aldığı Ürünler (Sebep)': sebep_str,
                # Güven ve lift'leri numeric tut, sonra formatla
                'Güven Skoru (%)': rule.get('confidence', 0) * 100,
                'Lift Değeri': rule.get('lift', 0)
            })

    if not potansiyel_tavsiyeler:
        return pd.DataFrame()

    tavsiyeler_df = pd.DataFrame(potansiyel_tavsiyeler)
    tavsiyeler_df = tavsiyeler_df.drop_duplicates(subset=['Tavsiye Edilen Ürün'])

    # kesin dönüşüm: eğer hala array/frozenset varsa stringe çevir
    for col in ['Tavsiye Edilen Ürün', 'Satın Aldığı Ürünler (Sebep)']:
        if col in tavsiyeler_df.columns:
            tavsiyeler_df[col] = tavsiyeler_df[col].apply(
                lambda x: ', '.join(map(str, x)) if isinstance(x, (list, tuple, set, frozenset)) else str(x)
            )

    # numeric kolonları düzgün sırala/formatla
    tavsiyeler_df['Güven Skoru (%)'] = tavsiyeler_df['Güven Skoru (%)'].astype(float)
    tavsiyeler_df['Lift Değeri'] = tavsiyeler_df['Lift Değeri'].astype(float)

    # sırala ve okunaklı formata çevir
    tavsiyeler_df = tavsiyeler_df.sort_values('Güven Skoru (%)', ascending=False)
    tavsiyeler_df['Güven Skoru (%)'] = tavsiyeler_df['Güven Skoru (%)'].map(lambda v: f"{v:.1f}%")
    tavsiyeler_df['Lift Değeri'] = tavsiyeler_df['Lift Değeri'].map(lambda v: f"{v:.2f}")

    return tavsiyeler_df

def churn_nedenlerini_analiz_et(model, feature_names):
    """
    Eğitilmiş Lojistik Regresyon modelinin katsayılarını analiz eder.
    Hangi özelliğin churn'ü ne kadar etkilediğini gösteren bir DataFrame döndürür.
    """
    if model is None:
        return pd.DataFrame()

    # Modelin katsayılarını (her bir özelliğin önemini) al
    katsayilar = model.coef_[0]
    
    # Katsayıları ve özellik isimlerini bir DataFrame'de birleştir
    etkenler_df = pd.DataFrame(data={'Özellik': feature_names, 'Katsayı': katsayilar})
    
    # Katsayının mutlak değerine göre sırala (en etkili olan en üstte)
    etkenler_df['Etki Büyüklüğü'] = etkenler_df['Katsayı'].abs()
    etkenler_df = etkenler_df.sort_values('Etki Büyüklüğü', ascending=False).drop(columns=['Etki Büyüklüğü'])
    
    # Odds Ratio'yu hesapla (yorumu kolaylaştırır)
    # 1'den büyükse churn riskini artırır, küçükse azaltır.
    etkenler_df['Odds Oranı'] = np.exp(etkenler_df['Katsayı'])
    
    return etkenler_df

def bireysel_churn_etkenlerini_hesapla(model, musteri_verileri, baseline_df):
    """
    Tek bir müşterinin churn skorunu hangi özelliğin ne kadar etkilediğini hesaplar.
    Müşteriyi, churn etmemiş 'güvenli' müşterilerin ortalamasıyla karşılaştırır.
    """
    if model is None:
        return pd.DataFrame()
    
    # Baseline: Churn etmemiş müşterilerin ortalama değerleri
    baseline = baseline_df[['Recency', 'Frequency', 'Monetary']].mean()
    
    # Müşterinin değerleri ile baseline arasındaki fark
    farklar = musteri_verileri[['Recency', 'Frequency', 'Monetary']] - baseline
    
    # Her bir farkın skora katkısı = fark * modelin katsayısı
    katsayilar = model.coef_[0]
    katkilar = farklar * katsayilar
    
    # Sonuçları bir DataFrame'e çevir
    katki_df = pd.DataFrame(katkilar).reset_index()
    katki_df.columns = ['Özellik', 'Katkı Skoru']
    
    # Modelin intercept'ini (başlangıç skoru) de ekleyelim
    intercept_df = pd.DataFrame([{'Özellik': 'Başlangıç (Ortalama Müşteri)', 'Katkı Skoru': model.intercept_[0]}])
    
    sonuc_df = pd.concat([intercept_df, katki_df], ignore_index=True)
    return sonuc_df

def ab_test_analizi_yap(df_a, df_b, metrik_kolonu, alpha=0.05):
    """
    İki farklı grup (A ve B) arasında belirli bir metrik için bağımsız T-testi yapar.
    Sonuçları, p-değeri ve yorumuyla birlikte bir sözlük olarak döndürür.
    """
    # Grupların verilerini ve eksik değerleri temizle
    grup_a_verisi = df_a[metrik_kolonu].dropna()
    grup_b_verisi = df_b[metrik_kolonu].dropna()
    
    # Varyansların homojenliğini test et (Levene Testi)
    levene_p_value = levene(grup_a_verisi, grup_b_verisi).pvalue
    varyanslar_esit = levene_p_value > alpha
    
    # T-testini uygula
    t_stat, p_value = ttest_ind(grup_a_verisi, grup_b_verisi, equal_var=varyanslar_esit)
    
    # Sonuçları yorumla
    if p_value < alpha:
        yorum = f"P-değeri ({p_value:.4f}) anlamlılık seviyesinden ({alpha}) küçük olduğu için, iki grup arasında istatistiksel olarak **anlamlı bir fark vardır.**"
    else:
        yorum = f"P-değeri ({p_value:.4f}) anlamlılık seviyesinden ({alpha}) büyük olduğu için, iki grup arasında istatistiksel olarak **anlamlı bir fark yoktur.**"
        
    sonuclar = {
        "grup_a_ort": grup_a_verisi.mean(),
        "grup_b_ort": grup_b_verisi.mean(),
        "fark_yuzdesi": ((grup_b_verisi.mean() - grup_a_verisi.mean()) / grup_a_verisi.mean()) * 100,
        "p_degeri": p_value,
        "yorum": yorum
    }
    
    return sonuclar

def pdf_raporu_olustur(musteri_id, musteri_verisi, forecast_fig, tavsiyeler_df):
    """GÜNCELLENMİŞ: Müşteri Detay Raporunu, doğru sütun sayılarıyla oluşturur."""
    pdf = PDF()
    # Fontlar PDF sınıfının __init__ metodunda otomatik olarak eklendiği için tekrar eklemeye gerek yok.
    pdf.add_page()
    
    pdf.chapter_title(f'Müşteri Analiz Raporu: {musteri_id}')
    
    pdf.set_font('DejaVu', 'B', 11)
    pdf.cell(0, 10, 'Genel Müşteri Metrikleri', 0, 1)
    pdf.set_font('DejaVu', '', 10)
    metrics = {
        'Segment': str(musteri_verisi['Segment']),
        'Performans Skoru (MPS)': f"{musteri_verisi['MPS']:.0f}",
        'Churn Olasılığı': f"{musteri_verisi['Churn_Olasiligi']*100:.1f}%",
        'Yaşam Boyu Değeri (CLV)': f"{musteri_verisi['CLV_Net_Kar']:,.0f} €"
    }
    for key, value in metrics.items():
        pdf.cell(60, 8, f'- {key}:', 0, 0)
        pdf.cell(0, 8, value, 0, 1)
    pdf.ln(5)

    if forecast_fig:
        pdf.chapter_title('Gelecek 6 Aylık Satış Tahmini')
        buf = io.BytesIO()
        forecast_fig.savefig(buf, format="PNG", dpi=200, bbox_inches='tight')
        buf.seek(0)
        pdf.image(buf, w=180)
        buf.close()
        pdf.ln(5)

    if not tavsiyeler_df.empty:
        if pdf.get_y() > 180:
             pdf.add_page()
             
        pdf.chapter_title('Ürün Önerileri (Next Best Offer)')
        
        # DataFrame'i PDF'e uygun hale getir (frozenset'leri metne çevir)
        tavsiye_listesi = tavsiyeler_df.head(15).copy()

        # Tablo verisini fpdf'in formatına uygun hale getir
        # Sütun başlıklarını alıyoruz
        table_data = [ tavsiyeler_df.columns.tolist() ] 
        # Satır verilerini alıyoruz
        def _to_str_cell(x):
            if isinstance(x, (list, tuple, set, frozenset)):
                return ', '.join(map(str, x))
            return str(x)

        for index, row in tavsiye_listesi.iterrows():
            antecedents_str = _to_str_cell(row.get('Satın Aldığı Ürünler (Sebep)', ''))
            consequents_str = _to_str_cell(row.get('Tavsiye Edilen Ürün', ''))
            table_data.append([consequents_str, antecedents_str, row.get('Güven Skoru (%)', ''), row.get('Lift Değeri', '')])


        pdf.set_font('DejaVu', '', 8)
        
        with pdf.table(
            col_widths=(45, 85, 30, 20),           # 4 sütun için genişlikler eklendi
            text_align=("LEFT", "LEFT", "RIGHT", "RIGHT"), # 4 sütun için hizalamalar eklendi
            headings_style=FontFace(emphasis="BOLD")
        ) as table:
            for data_row in table_data:
                row = table.row()
                for i, datum in enumerate(data_row):
                    # Sayısal değerleri formatlayalım
                    if isinstance(datum, float):
                        datum = f"{datum:.2f}"
                    row.cell(str(datum))
            
    return bytes(pdf.output())

def genel_rapor_pdf_olustur(filtrelenmis_df, baslangic_tarihi, bitis_tarihi, fig_pie, fig_bar):
    """GÜNCELLENMİŞ: Genel Bakış Raporunu daha düzenli ve sağlam tablolarla oluşturur."""
    pdf = PDF()
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
    pdf.add_page()
    
    pdf.chapter_title('Genel Bakış Analiz Raporu')
    pdf.chapter_body(f"Filtreleme Tarih Aralığı: {baslangic_tarihi.strftime('%d.%m.%Y')} - {bitis_tarihi.strftime('%d.%m.%Y')}")
    pdf.ln(5)

    pdf.chapter_title('Filtrelenmiş Veri KPIları')
    pdf.set_font('DejaVu', '', 10)
    kpis = {
        "Aktif Müşteri Sayısı": filtrelenmis_df.shape[0],
        "Bu Müşterilerin Toplam Cirosu": f"{filtrelenmis_df['Monetary'].sum():,.0f} €",
        "Ortalama Yaşam Boyu Değeri": f"{filtrelenmis_df['CLV_Net_Kar'].mean():,.0f} €",
        "Bu Gruptaki Şampiyon Sayısı": filtrelenmis_df[filtrelenmis_df['Segment'] == 'Şampiyonlar'].shape[0]
    }
    for key, value in kpis.items():
        pdf.cell(95, 8, f'- {key}:', 0, 0)
        pdf.cell(95, 8, str(value), 0, 1)
    pdf.ln(10)

    pdf.chapter_title('Görsel Analizler')
    if fig_pie: pdf.add_plotly_fig(fig_pie)
    pdf.ln(5)
    if fig_bar: pdf.add_plotly_fig(fig_bar)
    
    # Yeni sayfada tabloları göster
    if not filtrelenmis_df.empty:
        pdf.add_page()
        
        # --- DÜZELTİLMİŞ BÖLÜM: En Riskli Müşteriler Tablosu ---
        pdf.chapter_title('En Riskli 10 Müşteri (Churn Olasılığı Yüksek)')
        riskli_musteriler = filtrelenmis_df[['Segment', 'Recency', 'Churn_Olasiligi']].sort_values('Churn_Olasiligi', ascending=False).head(10)
        
        pdf.set_font('DejaVu', 'B', 9)
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(60, 8, 'Müşteri ID', 1, 0, 'C', 1)
        pdf.cell(50, 8, 'Segment', 1, 0, 'C', 1)
        pdf.cell(40, 8, 'Son Alım (Gün)', 1, 0, 'C', 1)
        pdf.cell(40, 8, 'Churn Olasılığı', 1, 1, 'C', 1)

        pdf.set_font('DejaVu', '', 8)
        for index, row in riskli_musteriler.iterrows():
            pdf.cell(60, 6, str(index), 1, 0)
            pdf.cell(50, 6, str(row['Segment']), 1, 0)
            pdf.cell(40, 6, str(row['Recency']), 1, 0, 'C')
            pdf.cell(40, 6, f"{row['Churn_Olasiligi']:.1%}", 1, 1, 'R')
        pdf.ln(10)
        
        # --- DÜZELTİLMİŞ BÖLÜM: En Değerli Müşteriler Tablosu ---
        pdf.chapter_title('En Değeri Yüksek 10 Müşteri (CLV''ye Göre)')
        degerli_musteriler = filtrelenmis_df[['Segment', 'CLV_Net_Kar']].sort_values(by='CLV_Net_Kar', ascending=False).head(10)

        pdf.set_font('DejaVu', 'B', 9)
        pdf.cell(90, 8, 'Müşteri ID', 1, 0, 'C', 1)
        pdf.cell(50, 8, 'Segment', 1, 0, 'C', 1)
        pdf.cell(50, 8, 'Tahmini Net Kar (CLV)', 1, 1, 'C', 1)

        pdf.set_font('DejaVu', '', 8)
        for index, row in degerli_musteriler.iterrows():
            pdf.cell(90, 6, str(index), 1, 0)
            pdf.cell(50, 6, str(row['Segment']), 1, 0)
            pdf.cell(50, 6, f"{row['CLV_Net_Kar']:,.0f} €", 1, 1, 'R')
            
    return bytes(pdf.output())

def sayfa_raporu_olustur(sayfa_basligi, fig=None, df=None):
    """
    GÜNCELLENMİŞ: Herhangi bir sayfa için daha düzenli tablolarla PDF raporu oluşturan genel fonksiyon.
    """
    pdf = PDF()
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
    pdf.add_page()
    
    pdf.chapter_title(sayfa_basligi)
    
    if fig:
        pdf.add_plotly_fig(fig)
        pdf.ln(5)
    
    if df is not None and not df.empty:
        pdf.chapter_title("Veri Tablosu")
        
        # DataFrame'i fpdf2'nin table() formatına uygun hale getir
        table_data = [df.columns.tolist()] + df.head(30).values.tolist()
        
        with pdf.table(
            text_align="LEFT", 
            headings_style=FontFace(emphasis="BOLD"),
            line_height=pdf.font_size * 1.5,
        ) as table:
            for data_row in table_data:
                row = table.row()
                for datum in data_row:
                    row.cell(str(datum))
            
    return bytes(pdf.output())

def kampanya_onerileri_uret(sonuclar_df):
    """
    Müşteri segmentasyon sonuçlarına göre her bir segment için
    pazarlama hedefleri, hedef kitle büyüklüğü ve özel kampanya fikirleri üretir.
    """
    # Segmentlerin var olup olmadığını kontrol et
    if 'Segment' not in sonuclar_df.columns:
        return {}

    segment_sayilari = sonuclar_df['Segment'].value_counts()
    
    oneriler = {
        'Şampiyonlar': {
            'Hedef': 'Ödüllendirme ve marka elçisi yapma.',
            'Hedef Kitle Büyüklüğü': segment_sayilari.get('Şampiyonlar', 0),
            'Kampanya Fikirleri': [
                "Yeni çıkacak ürünlere öncelikli erişim hakkı tanıyın.",
                "VIP müşteri etkinliklerine veya özel indirim günlerine davet edin.",
                "Markanız hakkında yorum yapmaları karşılığında sadakat puanı hediye edin."
            ]
        },
        'Potansiyel Şampiyonlar': {
            'Hedef': 'Satın alma sıklığını veya sepet tutarını artırarak "Şampiyon" segmentine taşımak.',
            'Hedef Kitle Büyüklüğü': segment_sayilari.get('Potansiyel Şampiyonlar', 0),
            'Kampanya Fikirleri': [
                "İlgilendikleri ürün kategorilerinde kişiselleştirilmiş teklifler sunun.",
                "Belirli bir sepet tutarının üzerine çıktıklarında özel indirimler tanımlayın.",
                "Çapraz satış (cross-sell) için ürün öneri motorunun sonuçlarını kullanın."
            ]
        },
        'Sadık Müşteriler': {
            'Hedef': 'Marka ile olan bağlarını güçlendirmek ve onları unutmadığınızı hissettirmek.',
            'Hedef Kitle Büyüklüğü': segment_sayilari.get('Sadık Müşteriler', 0),
            'Kampanya Fikirleri': [
                "Özel günlerde (doğum günü, üyelik yıldönümü) küçük hediyeler veya indirimler gönderin.",
                "Geri bildirimlerini ve fikirlerini almak için anketler düzenleyin.",
                "Sadece bu segmente özel, sınırlı süreli bir kampanya yapın."
            ]
        },
        'Riskli Müşteriler': {
            'Hedef': 'Müşteriyi kaybetmeden önce proaktif olarak geri kazanmak (reactivation).',
            'Hedef Kitle Büyüklüğü': segment_sayilari.get('Riskli Müşteriler', 0),
            'Kampanya Fikirleri': [
                "'Sizi Özledik!' temalı, kişiye özel ve cazip bir indirim kuponu sunun.",
                "Son satın aldığı ürünle ilgili tamamlayıcı bir üründe indirim teklif edin.",
                "Churn olasılığı en yüksek olanlara telefonla ulaşıp özel bir teklif sunun."
            ]
        },
        'Kayıp Müşteriler': {
            'Hedef': 'Son bir deneme ile geri kazanmak (win-back).',
            'Hedef Kitle Büyüklüğü': segment_sayilari.get('Kayıp Müşteriler', 0),
            'Kampanya Fikirleri': [
                "Geri çeviremeyecekleri, tek seferlik çok yüksek bir indirim ('Geri Dön' teklifi) sunun.",
                "Neden artık alışveriş yapmadıklarını anlamak için bir anket gönderin.",
                "Tamamen yeni bir ürün veya hizmet kategorisiyle ilgilerini çekmeye çalışın."
            ]
        }
    }
    
    return oneriler

def kampanya_roi_simulasyonu_yap(sonuclar_df, hedef_segment, beklenen_etki_orani, indirim_orani, musteri_basi_maliyet):
    """
    GÜNCELLENMİŞ: Ürün maliyetini (COGS) kullanıcıdan istemek yerine, doğrudan müşteri verisinden
    hesaplanan gerçek kar marjını kullanarak ROI simülasyonu yapar.
    """
    hedef_kitle_df = sonuclar_df[sonuclar_df['Segment'] == hedef_segment].copy()
    hedef_kitle_sayisi = len(hedef_kitle_df)

    if hedef_kitle_sayisi == 0:
        return {}

    # Adım 1: Segmentin gerçek kar marjını veriden hesapla
    toplam_ciro_gecmis = hedef_kitle_df['Monetary'].sum()
    toplam_kar_gecmis = hedef_kitle_df['ToplamNetKar'].sum()
    
    if toplam_ciro_gecmis > 0:
        gercek_kar_marji = toplam_kar_gecmis / toplam_ciro_gecmis
    else:
        gercek_kar_marji = 0.25 # Eğer segmentin geçmiş cirosu yoksa, varsayılan bir marj kullan

    # Adım 2: Kampanya etkisini simüle et
    hedef_kitle_df['baseline_alim_olasiligi'] = 1 - hedef_kitle_df['Churn_Olasiligi']
    hedef_kitle_df['kampanya_alim_olasiligi'] = np.minimum(1.0, hedef_kitle_df['baseline_alim_olasiligi'] * (1 + (beklenen_etki_orani / 100)))
    
    baseline_donusum = hedef_kitle_df['baseline_alim_olasiligi'].sum()
    kampanya_donusumu = hedef_kitle_df['kampanya_alim_olasiligi'].sum()
    ekstra_donusum_sayisi = kampanya_donusumu - baseline_donusum
    
    # Adım 3: Finansal sonuçları hesapla
    ortalama_sepet_tutari = hedef_kitle_df['Ortalama_Siparis_Tutari'].mean()
    tahmini_toplam_ciro = kampanya_donusumu * ortalama_sepet_tutari
    
    tahmini_brut_kar = tahmini_toplam_ciro * gercek_kar_marji
    
    sabit_iletisim_maliyeti = hedef_kitle_sayisi * musteri_basi_maliyet
    toplam_indirim_maliyeti = tahmini_toplam_ciro * (indirim_orani / 100)
    toplam_kampanya_maliyeti = sabit_iletisim_maliyeti + toplam_indirim_maliyeti
    
    tahmini_net_kar = tahmini_brut_kar - toplam_kampanya_maliyeti
    
    if toplam_kampanya_maliyeti > 0:
        roi = (tahmini_net_kar / toplam_kampanya_maliyeti) * 100
    else:
        roi = np.inf if tahmini_net_kar > 0 else 0

    return {
        "Hedef Kitle Sayısı": hedef_kitle_sayisi,
        "Tahmini Ekstra Müşteri": ekstra_donusum_sayisi,
        "Tahmini Toplam Ciro": tahmini_toplam_ciro,
        "Toplam Maliyet": toplam_kampanya_maliyeti,
        "Tahmini Net Kar": tahmini_net_kar,
        "Tahmini ROI (%)": roi
    }

def optimal_indirim_hesapla(sonuclar_df, hedef_segment, musteri_basi_maliyet, agirlik_kar=0.7, agirlik_etki=0.3):
    """
    SADELEŞTİRİLMİŞ: Kanal çarpanı olmadan optimizasyon yapar.
    """
    sonuclar = []
    indirim_oranlari = np.arange(1, 51, 1)
    
    def etki_hesapla(indirim):
        return 80 * np.log10(indirim + 1) / np.log10(51)

    for indirim in indirim_oranlari:
        beklenen_etki = etki_hesapla(indirim)
        
        simulasyon = kampanya_roi_simulasyonu_yap(
            sonuclar_df, 
            hedef_segment, 
            beklenen_etki, 
            indirim, 
            musteri_basi_maliyet
        )
        if simulasyon:
            sonuclar.append({
                "İndirim Oranı (%)": indirim,
                "Tahmini Net Kar (€)": simulasyon['Tahmini Net Kar'],
                "Beklenen Etki (%)": beklenen_etki
            })
            
    if not sonuclar: 
        return pd.DataFrame(), None
        
    opt_df = pd.DataFrame(sonuclar)
    
    kar_min, kar_max = opt_df['Tahmini Net Kar (€)'].min(), opt_df['Tahmini Net Kar (€)'].max()
    etki_min, etki_max = opt_df['Beklenen Etki (%)'].min(), opt_df['Beklenen Etki (%)'].max()
    
    if pd.isna(kar_max) or pd.isna(kar_min):
        return pd.DataFrame(), None
    
    if (kar_max - kar_min) > 0:
        opt_df['Kar_Normalized'] = (opt_df['Tahmini Net Kar (€)'] - kar_min) / (kar_max - kar_min)
    else:
        opt_df['Kar_Normalized'] = 1.0

    if (etki_max - etki_min) > 0:
        opt_df['Etki_Normalized'] = (opt_df['Beklenen Etki (%)'] - etki_min) / (etki_max - etki_min)
    else:
        opt_df['Etki_Normalized'] = 1.0
        
    opt_df['Optimizasyon_Skoru'] = (agirlik_kar * opt_df['Kar_Normalized']) + (agirlik_etki * opt_df['Etki_Normalized'])
    
    optimal_nokta = opt_df.loc[opt_df['Optimizasyon_Skoru'].idxmax()]
    
    return opt_df, optimal_nokta

def kampanya_basari_analizi_yap(temiz_df, hedef_musteri_idler, kampanya_baslangic_tarihi, takip_suresi_gun):
    """
    Gerçekleşmiş bir kampanyanın sonuçlarını analiz eder.
    Hedeflenen müşterilerden kaçının kampanya döneminde alım yaptığını (dönüşüm) hesaplar.
    """
    kampanya_bitis_tarihi = kampanya_baslangic_tarihi + timedelta(days=takip_suresi_gun)
    hedef_musteri_sayisi = len(hedef_musteri_idler)
    
    # Hedeflenen müşterilerin kampanya dönemindeki alımlarını bul
    donusum_df = temiz_df[
        (temiz_df['MusteriID'].isin(hedef_musteri_idler)) &
        (temiz_df['Tarih'] >= pd.to_datetime(kampanya_baslangic_tarihi)) &
        (temiz_df['Tarih'] <= pd.to_datetime(kampanya_bitis_tarihi))
    ]
    
    donusum_yapan_musteriler = donusum_df['MusteriID'].unique()
    donusum_sayisi = len(donusum_yapan_musteriler)
    donusum_orani = (donusum_sayisi / hedef_musteri_sayisi) * 100 if hedef_musteri_sayisi > 0 else 0
    
    toplam_ciro = donusum_df['ToplamTutar'].sum()
    
    return {
        "Hedeflenen Müşteri Sayısı": hedef_musteri_sayisi,
        "Dönüşüm Yapan Müşteri Sayısı": donusum_sayisi,
        "Dönüşüm Oranı (%)": donusum_orani,
        "Kampanyadan Gelen Toplam Ciro": toplam_ciro
    }

def roi_simulasyon_raporu_pdf_olustur(sonuclar, segment, maliyet, indirim, etki):
    """SADELEŞTİRİLMİŞ: Kanal bilgisi olmadan PDF raporu oluşturur."""
    pdf = PDF()
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
    pdf.add_page()
    
    pdf.chapter_title(f"ROI Simülasyon Raporu: '{segment}' Segmenti")
    
    pdf.set_font('DejaVu', 'B', 11)
    pdf.cell(0, 10, 'Kampanya Parametreleri', 0, 1)
    pdf.set_font('DejaVu', '', 10)
    
    params = {
        "Müşteri Başına Maliyet": f"{maliyet:.2f} €",
        "Uygulanan İndirim": f"%{indirim}",
        "Beklenen Etki (Alım Olasılığı Artışı)": f"%{etki:.1f}"
    }
    for key, value in params.items():
        pdf.cell(95, 8, f'- {key}:', 0, 0)
        pdf.cell(95, 8, str(value), 0, 1)
    pdf.ln(10)

    pdf.chapter_title('Simülasyon Sonuçları')
    pdf.set_font('DejaVu', '', 10)
    
    for key, value in sonuclar.items():
        formatted_value = f"{value:,.2f}" if isinstance(value, (int, float)) else str(value)
        if "€" in key or "%" in key:
            formatted_value += f" {key.split('(')[-1].split(')')[0]}"
        pdf.cell(95, 8, f'- {key.split("(")[0].strip()}:', 0, 0)
        pdf.cell(95, 8, formatted_value, 0, 1)
        
    return bytes(pdf.output())

def optimal_indirim_raporu_pdf_olustur(optimal_nokta, fig_opt):
    """YENİ: Optimal İndirim Analizi sekmesi için özel PDF raporu oluşturur."""
    pdf = PDF()
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
    pdf.add_page()
    
    pdf.chapter_title("Optimal İndirim Analizi Raporu")
    
    pdf.set_font('DejaVu', 'B', 11)
    pdf.cell(0, 10, 'Optimizasyon Sonuçları', 0, 1)
    pdf.set_font('DejaVu', '', 10)
    
    metrics = {
        "Stratejiye Göre Optimal İndirim Oranı": f"%{optimal_nokta['İndirim Oranı (%)']:.0f}",
        "Maksimum Tahmini Net Kar": f"{optimal_nokta['Tahmini Net Kar (€)']:,.0f} €"
    }
    for key, value in metrics.items():
        pdf.cell(95, 8, f'- {key}:', 0, 0)
        pdf.cell(95, 8, str(value), 0, 1)
    pdf.ln(10)
    
    pdf.chapter_title("İndirim-Kar İlişkisi Grafiği")
    if fig_opt:
        pdf.add_plotly_fig(fig_opt)
        
    return bytes(pdf.output())

def anomali_tespiti_yap(sonuclar_df, kontaminasyon_orani=0.05):
    """
    Müşteri RFM verileri üzerinde Isolation Forest algoritmasını kullanarak anomali tespiti yapar.
    """
    # Analiz için kullanılacak özellikleri seçelim
    ozellikler = sonuclar_df[['Recency', 'Frequency', 'Monetary']].copy()
    
    # Modeli kuralım
    # contamination: Veri setindeki beklenen anomali oranıdır.
    model = IsolationForest(contamination=kontaminasyon_orani, random_state=42)
    
    # Modeli eğitelim ve tahmin yapalım
    model.fit(ozellikler)
    
    # Sonuçları ana DataFrame'e ekleyelim
    # decision_function: Bir değerin ne kadar 'anormal' olduğunu gösteren bir skor. Negatif değerler anomaliye daha yakındır.
    sonuclar_df['Anomali_Skoru'] = model.decision_function(ozellikler)
    # predict: Her müşteri için etiket atar. -1 = Anomali, 1 = Normal.
    sonuclar_df['Anomali_Etiketi'] = model.predict(ozellikler)
    
    return sonuclar_df.sort_values('Anomali_Skoru')

def davranissal_anomali_tespiti_yap(temiz_df, hassasiyet=2.5):
    """
    Her müşterinin kendi geçmiş satın alma sıklığını analiz eder.
    Son alım tarihini ve o tarihten bugüne geçen süreyi de sonuçlara ekler.
    """
    islem_sayilari = temiz_df['MusteriID'].value_counts()
    yeterli_verisi_olan_musteriler = islem_sayilari[islem_sayilari >= 5].index
    
    df_analiz = temiz_df[temiz_df['MusteriID'].isin(yeterli_verisi_olan_musteriler)].copy()
    df_analiz = df_analiz.sort_values(['MusteriID', 'Tarih'])
    
    df_analiz['Alimlar_Arasi_Sure'] = df_analiz.groupby('MusteriID')['Tarih'].diff().dt.days
    
    istatistikler = df_analiz.groupby('MusteriID')['Alimlar_Arasi_Sure'].agg(['mean', 'std']).reset_index()
    istatistikler.rename(columns={'mean': 'Ortalama_Gun', 'std': 'Std_Sapma'}, inplace=True)
    istatistikler.fillna(0, inplace=True)

    son_alim_araliklari = df_analiz.groupby('MusteriID')['Alimlar_Arasi_Sure'].last().reset_index()
    son_alim_araliklari.rename(columns={'Alimlar_Arasi_Sure': 'Son_Alim_Araligi'}, inplace=True)
    
    sonuclar = pd.merge(istatistikler, son_alim_araliklari, on='MusteriID')
    
    sonuclar['Anomali_Esigi_Gun'] = sonuclar['Ortalama_Gun'] + (hassasiyet * sonuclar['Std_Sapma'])
    
    sonuclar['Anomali_Durumu'] = sonuclar.apply(
        lambda row: "Evet" if row['Son_Alim_Araligi'] > row['Anomali_Esigi_Gun'] and row['Anomali_Esigi_Gun'] > 0 else "Hayır", 
        axis=1
    )
    
    anomaliler_df = sonuclar[sonuclar['Anomali_Durumu'] == 'Evet']
    
    if anomaliler_df.empty:
        return pd.DataFrame()

    son_alim_tarihleri = temiz_df.groupby('MusteriID')['Tarih'].max().reset_index()
    son_alim_tarihleri.rename(columns={'Tarih': 'Son_Alim_Tarihi'}, inplace=True)
    
    anomaliler_df = pd.merge(anomaliler_df, son_alim_tarihleri, on='MusteriID')

    # Analizin yapıldığı "bugün" tarihini verideki en son tarih olarak alalım
    analiz_gunu = temiz_df['Tarih'].max()
    # Son alım tarihinden bugüne geçen gün sayısını hesaplayalım
    anomaliler_df['Gecen_Sure_Gun'] = (analiz_gunu - anomaliler_df['Son_Alim_Tarihi']).dt.days

    return anomaliler_df.sort_values('Son_Alim_Araligi', ascending=False)

def anomali_tespiti_dbscan(sonuclar_df, eps=0.5, min_samples=5):
    """
    Müşteri RFM verileri üzerinde DBSCAN algoritmasını kullanarak anomali tespiti yapar.
    Veriyi analizden önce ölçeklendirir.
    """
    # Analiz için kullanılacak özellikleri seçelim
    ozellikler = sonuclar_df[['Recency', 'Frequency', 'Monetary']].copy()
    
    # DBSCAN mesafeye dayalı olduğu için veriyi ölçeklendirmek çok önemlidir.
    scaler = StandardScaler()
    ozellikler_scaled = scaler.fit_transform(ozellikler)
    
    # Modeli kuralım ve eğitelim
    # eps: Bir noktanın komşuluğunu tanımlayan mesafe.
    # min_samples: Yoğun bir bölge oluşturmak için gereken minimum komşu sayısı.
    model = DBSCAN(eps=eps, min_samples=min_samples)
    
    # Kümeleri ve anomalileri tahmin et
    tahminler = model.fit_predict(ozellikler_scaled)
    
    # Sonuçları ana DataFrame'e ekleyelim
    # DBSCAN, anomalileri -1 olarak etiketler.
    sonuclar_df['Anomali_Etiketi'] = tahminler
    
    return sonuclar_df.sort_values('Anomali_Etiketi') # Anomaliler (-1) en üste gelecek

def islem_bazli_anomali_tespiti_yap(temiz_df, kontaminasyon_orani=0.01):
    """
    Her bir satış işlemini analiz ederek aykırı olanları bulur.
    Miktar, ToplamTutar ve işlem saati gibi özellikleri kullanır.
    """
    df_kopya = temiz_df.copy()
    
    # Yeni özellikler türetelim: İşlemin yapıldığı saat ve haftanın günü
    df_kopya['Saat'] = df_kopya['Tarih'].dt.hour
    df_kopya['Haftanin_Gunu'] = df_kopya['Tarih'].dt.dayofweek # Pazartesi=0, Pazar=6
    
    # Modelin kullanacağı özellikleri seçelim
    ozellikler = df_kopya[['Miktar', 'BirimFiyat', 'ToplamTutar', 'Saat', 'Haftanin_Gunu']]
    
    # Isolation Forest modelini kuralım
    model = IsolationForest(contamination=kontaminasyon_orani, random_state=42)
    model.fit(ozellikler)
    
    # Tahminleri yapıp ana DataFrame'e ekleyelim
    df_kopya['Anomali_Skoru'] = model.decision_function(ozellikler)
    df_kopya['Anomali_Etiketi'] = model.predict(ozellikler)
    
    # Sadece anomali olarak etiketlenen işlemleri (-1) döndürelim
    anomaliler_df = df_kopya[df_kopya['Anomali_Etiketi'] == -1].sort_values('Anomali_Skoru')
    
    return anomaliler_df

def anomali_gruplama_yap(anomaliler_df, kume_sayisi=3):
    """
    Tespit edilen anormal müşterileri, RFM profillerine göre K-Means ile kümelere ayırır
    ve bu kümeleri yorumlayarak etiketler.
    """
    if anomaliler_df.empty or len(anomaliler_df) < kume_sayisi:
        return pd.DataFrame(), pd.DataFrame()

    ozellikler = anomaliler_df[['Recency', 'Frequency', 'Monetary']].copy()
    
    # K-Means mesafeye dayalı olduğu için veriyi ölçeklendirmek çok önemlidir.
    scaler = StandardScaler()
    ozellikler_scaled = scaler.fit_transform(ozellikler)
    
    # K-Means modelini kur ve çalıştır
    kmeans = KMeans(n_clusters=kume_sayisi, init='k-means++', random_state=42, n_init='auto')
    anomaliler_df['Anomali_Grubu_ID'] = kmeans.fit_predict(ozellikler_scaled)
    
    # Kümelerin merkezlerini (ortalama profillerini) bul ve orijinal ölçeğe geri çevir
    merkezler_scaled = kmeans.cluster_centers_
    merkezler = scaler.inverse_transform(merkezler_scaled)
    merkezler_df = pd.DataFrame(merkezler, columns=['Recency', 'Frequency', 'Monetary'])
    
    # Merkezleri yorumlayarak her bir gruba anlamlı bir isim verelim
    grup_isimleri = {}
    for i, row in merkezler_df.iterrows():
        # En baskın özelliği bularak gruba isim ver
        if row['Monetary'] > merkezler_df['Monetary'].mean() * 1.5 and row['Frequency'] < merkezler_df['Frequency'].mean() * 0.8:
            grup_isimleri[i] = "Tek Seferlik Dev Alıcı (Whale)"
        elif row['Recency'] > merkezler_df['Recency'].mean() * 1.5:
            grup_isimleri[i] = "Terk Edilmiş Değerli Müşteri"
        elif row['Frequency'] > merkezler_df['Frequency'].mean() * 1.5:
             grup_isimleri[i] = "Aşırı Aktif Müşteri"
        else:
            grup_isimleri[i] = f"Anomali Grubu {i+1}"
            
    merkezler_df['Grup_Adi'] = merkezler_df.index.map(grup_isimleri)
    anomaliler_df['Anomali_Grubu'] = anomaliler_df['Anomali_Grubu_ID'].map(grup_isimleri)
    
    return anomaliler_df, merkezler_df.set_index('Grup_Adi')

def musteri_benzerlik_hesapla(sonuclar_df):
    """
    Müşterilerin RFM profillerini kullanarak aralarındaki benzerliği hesaplar.
    Sonuç olarak müşteri-müşteri benzerlik matrisini döndürür.
    """
    # Benzerlik için kullanılacak özellikleri seçelim
    ozellikler = sonuclar_df[['Recency', 'Frequency', 'Monetary']].copy()
    
    # Farklı ölçeklerdeki özellikleri standartlaştıralım
    scaler = StandardScaler()
    ozellikler_scaled = scaler.fit_transform(ozellikler)
    
    # Kosinüs benzerlik matrisini hesapla
    similarity_matrix = cosine_similarity(ozellikler_scaled)
    
    # Matrisi daha kullanışlı bir DataFrame'e çevirelim
    similarity_df = pd.DataFrame(similarity_matrix, index=sonuclar_df.index, columns=sonuclar_df.index)
    
    return similarity_df

def benzer_musteri_urun_onerileri(temiz_df, kaynak_musteri_id, benzer_musteri_idler):
    """
    Bir kaynak müşteriye en çok benzeyen müşterilerin satın aldığı, ancak kaynak
    müşterinin henüz almadığı ürünleri tespit eder ve öneri olarak listeler.
    """
    if not benzer_musteri_idler:
        return pd.DataFrame()

    # 1. Kaynak müşterinin zaten satın aldığı ürünleri bul
    kaynak_urunler = set(temiz_df[temiz_df['MusteriID'] == kaynak_musteri_id]['UrunKodu'].unique())
    
    # 2. Benzer müşterilerin tüm alımlarını filtrele
    benzerlerin_alimlari = temiz_df[temiz_df['MusteriID'].isin(benzer_musteri_idler)]
    
    # 3. Benzerlerin aldığı ama kaynak müşterinin almadığı ürünleri bul
    onerilecek_urunler = benzerlerin_alimlari[~benzerlerin_alimlari['UrunKodu'].isin(kaynak_urunler)]
    
    if onerilecek_urunler.empty:
        return pd.DataFrame()
        
    # 4. Bu öneri ürünlerini, benzer müşteriler arasındaki popülerliğine göre sırala
    oneri_skorlari = onerilecek_urunler.groupby('UrunKodu')['MusteriID'].nunique().reset_index()
    oneri_skorlari.rename(columns={'MusteriID': 'Benzer_Musteri_Sayisi'}, inplace=True)
    
    # Alım oranını hesapla
    oneri_skorlari['Alim_Orani'] = (oneri_skorlari['Benzer_Musteri_Sayisi'] / len(benzer_musteri_idler)) * 100
    
    return oneri_skorlari.sort_values('Benzer_Musteri_Sayisi', ascending=False)

def kmeans_kumeleme_yap(sonuclar_df, kume_sayisi=4, ozellikler=['Recency', 'Frequency', 'Monetary']):
    """
    GÜNCELLENMİŞ: Belirtilen özelliklere göre K-Means kümelemesi yapar.
    """
    df_ozellikler = sonuclar_df[ozellikler].copy()
    scaler = StandardScaler()
    ozellikler_scaled = scaler.fit_transform(df_ozellikler)
    
    kmeans = KMeans(n_clusters=kume_sayisi, init='k-means++', random_state=42, n_init='auto')
    sonuclar_df['Kume'] = kmeans.fit_predict(ozellikler_scaled)
    
    merkezler = scaler.inverse_transform(kmeans.cluster_centers_)
    merkezler_df = pd.DataFrame(merkezler, columns=ozellikler)
    
    return sonuclar_df, merkezler_df

def hiyerarsik_kumeleme_yap(sonuclar_df, kume_sayisi=4, ozellikler=['Recency', 'Frequency', 'Monetary']):
    """
    GÜNCELLENMİŞ: Belirtilen özelliklere göre Hiyerarşik Kümeleme yapar.
    """
    df_ozellikler = sonuclar_df[ozellikler].copy()
    scaler = StandardScaler()
    ozellikler_scaled = scaler.fit_transform(df_ozellikler)
    
    hiyerarsik = AgglomerativeClustering(n_clusters=kume_sayisi)
    sonuclar_df['Kume'] = hiyerarsik.fit_predict(ozellikler_scaled)
    
    merkezler_df = sonuclar_df.groupby('Kume')[ozellikler].mean()
    
    return sonuclar_df, merkezler_df

def pca_ile_boyut_indirge(sonuclar_df, ozellikler=['Recency', 'Frequency', 'Monetary']):
    """
    GÜNCELLENMİŞ: Belirtilen özelliklere göre veriyi 2 boyuta indirger.
    """
    df_ozellikler = sonuclar_df[ozellikler].copy()
    scaler = StandardScaler()
    ozellikler_scaled = scaler.fit_transform(df_ozellikler)
    
    pca = PCA(n_components=2)
    pca_sonuclari = pca.fit_transform(ozellikler_scaled)
    
    sonuclar_df['pca1'] = pca_sonuclari[:, 0]
    sonuclar_df['pca2'] = pca_sonuclari[:, 1]
    return sonuclar_df

def en_iyi_kume_sayisini_bul(sonuclar_df, max_kume=10, ozellikler=['Recency', 'Frequency', 'Monetary']):
    """
    GÜNCELLENMİŞ: Belirtilen özelliklere göre en iyi küme sayısını bulur.
    """
    df_ozellikler = sonuclar_df[ozellikler].copy()
    scaler = StandardScaler()
    ozellikler_scaled = scaler.fit_transform(df_ozellikler)
    
    skorlar = []
    for k in range(2, max_kume + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init='auto')
        kumeler = kmeans.fit_predict(ozellikler_scaled)
        skor = silhouette_score(ozellikler_scaled, kumeler)
        skorlar.append({'Küme Sayısı': k, 'Siluet Skoru': skor})
        
    return pd.DataFrame(skorlar)

def donemsel_analiz_yap(temiz_df, baslangic1, bitis1, baslangic2, bitis2):
    """
    Kullanıcı tarafından seçilen iki farklı zaman periyodu için temel metrikleri hesaplar ve
    karşılaştırma için bir sonuç sözlüğü döndürür.
    """
    # Veriyi periyotlara göre filtrele
    periyot1_df = temiz_df[
        (temiz_df['Tarih'].dt.date >= baslangic1) & 
        (temiz_df['Tarih'].dt.date <= bitis1)
    ]
    periyot2_df = temiz_df[
        (temiz_df['Tarih'].dt.date >= baslangic2) & 
        (temiz_df['Tarih'].dt.date <= bitis2)
    ]

    # Her müşterinin ilk alışveriş tarihini bul (yeni müşteri tespiti için)
    ilk_alisveris_tarihleri = temiz_df.groupby('MusteriID')['Tarih'].min().dt.date

    def metrikleri_hesapla(df, baslangic, bitis):
        toplam_ciro = df['ToplamTutar'].sum()
        aktif_musteri_sayisi = df['MusteriID'].nunique()
        islem_sayisi = len(df)
        
        # O dönemde kazanılan yeni müşterileri bul
        yeni_musteri_idler = ilk_alisveris_tarihleri[
            (ilk_alisveris_tarihleri >= baslangic) & 
            (ilk_alisveris_tarihleri <= bitis)
        ].index
        yeni_musteri_sayisi = len(yeni_musteri_idler)
        
        return {
            'Toplam Ciro': toplam_ciro,
            'Aktif Müşteri Sayısı': aktif_musteri_sayisi,
            'Ortalama Sepet Tutarı': toplam_ciro / islem_sayisi if islem_sayisi > 0 else 0,
            'Yeni Müşteri Sayısı': yeni_musteri_sayisi
        }

    sonuclar = {
        'Periyot 1': metrikleri_hesapla(periyot1_df, baslangic1, bitis1),
        'Periyot 2': metrikleri_hesapla(periyot2_df, baslangic2, bitis2)
    }
    return sonuclar

def benchmark_profili_hesapla(sonuclar_df, benchmark_tipi='Tüm Müşteriler'):
    """
    Belirtilen bir benchmark tipine göre (örn: 'Tüm Müşteriler' veya 'Şampiyonlar')
    ortalama müşteri profilini hesaplar.
    """
    benchmark_df = sonuclar_df
    # Eğer belirli bir segment seçildiyse, veriyi o segmente göre filtrele
    if benchmark_tipi != 'Tüm Müşteriler':
        benchmark_df = sonuclar_df[sonuclar_df['Segment'] == benchmark_tipi]

    if benchmark_df.empty:
        # Eğer o segmentte müşteri yoksa boş bir seri döndür
        return pd.Series(name=f"Ortalama '{benchmark_tipi}' Profili")

    # Ortalama alınacak metrikleri belirle
    metrikler = ['Recency', 'Frequency', 'Monetary', 'MPS', 'CLV_Net_Kar', 'Churn_Olasiligi', 'R_Score', 'F_Score', 'M_Score']
    mevcut_metrikler = [m for m in metrikler if m in benchmark_df.columns]
    
    ortalama_profil = benchmark_df[mevcut_metrikler].mean()
    ortalama_profil.name = f"Ortalama '{benchmark_tipi}' Profili"
    
    # Segment adını da ekleyelim
    ortalama_profil['Segment'] = benchmark_tipi
    
    return ortalama_profil

def zaman_serisi_ayristirma_yap(time_series_df, model_tipi='additive'):
    """
    Bir zaman serisini, statsmodels kullanarak trend, mevsimsellik ve kalıntı (residual)
    bileşenlerine ayırır. Multiplicative model için sıfır kontrolü eklendi.
    """
    # Analiz için tarih kolonunu index yapmalıyız
    df_ayristirma = time_series_df.set_index('ds')
    
    # --- YENİ GÜVENLİK KONTROLÜ ---
    # Eğer model çarpımsalsa ve veride 0 veya negatif değer varsa, hata döndür.
    if model_tipi == 'multiplicative' and (df_ayristirma['y'] <= 0).any():
        return None, "Çarpımsal (multiplicative) model, sıfır veya negatif ciro içeren aylar olduğu için kullanılamaz. Lütfen Toplamsal (additive) modeli deneyin."
    # --- GÜVENLİK KONTROLÜ SONU ---
    
    # period=12 -> yıllık mevsimsellik olduğunu varsayıyoruz (aylık veri için)
    try:
        ayristirma_sonucu = seasonal_decompose(df_ayristirma['y'], model=model_tipi, period=12)
        return ayristirma_sonucu, None # Başarılı olursa hata mesajı None döner
    except Exception as e:
        return None, f"Ayrıştırma sırasında bir hata oluştu: {e}"

def gelecek_tahmini_yap(time_series_df, tahmin_periyodu_ay=12):
    """
    Prophet kütüphanesini kullanarak, bir zaman serisi için gelecek tahmini yapar.
    GÜNCELLENMİŞ: Negatif tahmin sonuçlarını sıfırlar.
    """
    # Prophet modelini Türkiye tatillerini içerecek şekilde kur
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.add_country_holidays(country_name='TR')
    
    # Modeli eğit
    model.fit(time_series_df)
    
    # Gelecek için bir dataframe oluştur
    future = model.make_future_dataframe(periods=tahmin_periyodu_ay, freq='M')
    forecast = model.predict(future)
    forecast['yhat'] = np.maximum(0, forecast['yhat'])
    forecast['yhat_lower'] = np.maximum(0, forecast['yhat_lower'])
    forecast['yhat_upper'] = np.maximum(0, forecast['yhat_upper'])
    # --- DÜZELTME SONU ---

    return model, forecast

def kategori_migrasyon_analizi_yap(temiz_df):
    """
    Müşterilerin ilk ve ikinci satın alma kategorileri arasındaki geçişleri
    analiz eder ve bir geçiş matrisi döndürür.
    """
    # Analiz için sadece en az 2 farklı işlemi olan müşterileri al
    islem_sayilari = temiz_df['MusteriID'].value_counts()
    hedef_musteriler = islem_sayilari[islem_sayilari >= 2].index
    df_analiz = temiz_df[temiz_df['MusteriID'].isin(hedef_musteriler)].copy()
    
    df_analiz = df_analiz.sort_values(['MusteriID', 'Tarih'])
    
    # Her müşterinin ilk ve ikinci alımını bul
    ilk_alimlar = df_analiz.groupby('MusteriID').first().reset_index()
    ikinci_alimlar = df_analiz.groupby('MusteriID').nth(1).reset_index()
    
    # Geçiş verisini oluştur
    gecis_df = pd.merge(
        ilk_alimlar[['MusteriID', 'Kategori']],
        ikinci_alimlar[['MusteriID', 'Kategori']],
        on='MusteriID',
        suffixes=('_ilk', '_ikinci')
    )

    # Geçiş matrisini (crosstab) oluştur
    migrasyon_matrisi = pd.crosstab(gecis_df['Kategori_ilk'], gecis_df['Kategori_ikinci'], normalize='index')
    
    return migrasyon_matrisi

def kategori_performans_analizi_yap(temiz_df):
    """
    Her bir ürün kategorisinin performansını çeşitli metrikler üzerinden analiz eder.
    """
    if 'Kategori' not in temiz_df.columns:
        return pd.DataFrame()

    # Kategori bazında temel metrikleri hesapla
    kategori_performans = temiz_df.groupby('Kategori').agg(
        Toplam_Ciro=('ToplamTutar', 'sum'),
        Toplam_Net_Kar=('NetKar', 'sum'),
        Benzersiz_Musteri_Sayisi=('MusteriID', 'nunique'),
        Islem_Sayisi=('UrunKodu', 'count'), # UrunKodu count'u işlem sayısını verir
        Urun_Cesitliligi=('UrunKodu', 'nunique')
    ).reset_index()

    # Ek metrikleri hesapla
    kategori_performans['Kar_Marji'] = (kategori_performans['Toplam_Net_Kar'] / kategori_performans['Toplam_Ciro']) * 100
    kategori_performans['Musteri_Basina_Ciro'] = kategori_performans['Toplam_Ciro'] / kategori_performans['Benzersiz_Musteri_Sayisi']
    
    # NaN değerleri (0'a bölünme durumunda) temizle
    kategori_performans.fillna(0, inplace=True)
    
    return kategori_performans.sort_values('Toplam_Ciro', ascending=False)

def kategori_kannibalizasyon_analizi(temiz_df, kaynak_kategori, hedef_kategori, periyot_uzunlugu_ay=6):
    """
    İki kategori arasındaki olası bir kannibalizasyonu analiz eder.
    Kaynak kategoriden alım yapmayı bırakıp hedef kategoriye geçen müşterileri bulur
    ve bu geçişin finansal etkisini hesaplar.
    """
    if 'Kategori' not in temiz_df.columns:
        return None, "Veri setinde 'Kategori' sütunu bulunamadı."

    # Analiz periyotlarını belirle: son 6 ay ve önceki 6 ay
    analiz_gunu = temiz_df['Tarih'].max()
    periyot2_bitis = analiz_gunu
    periyot2_baslangic = analiz_gunu - pd.DateOffset(months=periyot_uzunlugu_ay)
    periyot1_bitis = periyot2_baslangic - pd.DateOffset(days=1)
    periyot1_baslangic = periyot1_bitis - pd.DateOffset(months=periyot_uzunlugu_ay)

    # Her periyottaki alımları filtrele
    periyot1_df = temiz_df[(temiz_df['Tarih'] >= periyot1_baslangic) & (temiz_df['Tarih'] <= periyot1_bitis)]
    periyot2_df = temiz_df[(temiz_df['Tarih'] >= periyot2_baslangic) & (temiz_df['Tarih'] <= periyot2_bitis)]

    # 1. Adım: İlk periyotta kaynak kategoriden alım yapan müşterileri bul
    kaynak_musteriler_p1 = set(periyot1_df[periyot1_df['Kategori'] == kaynak_kategori]['MusteriID'].unique())

    if not kaynak_musteriler_p1:
        return None, f"İlk periyotta '{kaynak_kategori}' kategorisinden hiç alım yapılmamış."

    # 2. Adım: Bu müşterilerden, ikinci periyotta kaynak kategoriden HİÇ alım yapmayanları bul (terk edenler)
    kaynak_musteriler_p2 = set(periyot2_df[periyot2_df['Kategori'] == kaynak_kategori]['MusteriID'].unique())
    terk_eden_musteriler = kaynak_musteriler_p1 - kaynak_musteriler_p2
    
    # 3. Adım: Terk eden bu müşterilerden, ikinci periyotta hedef kategoriye GEÇENLERİ bul
    hedef_musteriler_p2 = set(periyot2_df[periyot2_df['Kategori'] == hedef_kategori]['MusteriID'].unique())
    gecis_yapan_musteriler = list(terk_eden_musteriler.intersection(hedef_musteriler_p2))
    
    if not gecis_yapan_musteriler:
        return pd.DataFrame(), "Belirtilen periyotlarda bu iki kategori arasında bir geçiş tespit edilmedi."
        
    # 4. Adım: Finansal etkiyi hesapla
    # Kaybedilen ciro: Geçiş yapan müşterilerin ilk periyottaki kaynak kategori harcaması
    kaybedilen_ciro = periyot1_df[
        (periyot1_df['MusteriID'].isin(gecis_yapan_musteriler)) &
        (periyot1_df['Kategori'] == kaynak_kategori)
    ]['ToplamTutar'].sum()

    # Kazanılan ciro: Geçiş yapan müşterilerin ikinci periyottaki hedef kategori harcaması
    kazanilan_ciro = periyot2_df[
        (periyot2_df['MusteriID'].isin(gecis_yapan_musteriler)) &
        (periyot2_df['Kategori'] == hedef_kategori)
    ]['ToplamTutar'].sum()
    
    net_etki = kazanilan_ciro - kaybedilen_ciro
    
    sonuclar = {
        "Geçiş Yapan Müşteri Sayısı": len(gecis_yapan_musteriler),
        "Kaybedilen Ciro (Kaynak Kategoriden)": kaybedilen_ciro,
        "Kazanılan Ciro (Hedef Kategoriden)": kazanilan_ciro,
        "Net Ciro Etkisi": net_etki
    }
    
    # Geçiş yapan müşterilerin listesini de döndür
    gecis_yapan_df = temiz_df[temiz_df['MusteriID'].isin(gecis_yapan_musteriler)]

    return gecis_yapan_df, sonuclar

def otomatik_kannibalizasyon_bul(temiz_df, periyot_uzunlugu_ay=6):
    """
    Tüm olası kategori çiftleri arasında müşteri geçişlerini analiz eder ve
    en çok geçiş yaşanan veya en yüksek ciro etkisine sahip olanları bulur.
    """
    if 'Kategori' not in temiz_df.columns:
        return pd.DataFrame()

    kategoriler = temiz_df['Kategori'].unique()
    # Tüm olası (kaynak, hedef) çiftlerini oluştur
    kategori_ciftleri = list(permutations(kategoriler, 2))
    
    tum_sonuclar = []
    
    # Her bir çift için kannibalizasyon analizini çalıştır
    for kaynak, hedef in kategori_ciftleri:
        _, sonuc = kategori_kannibalizasyon_analizi(temiz_df, kaynak, hedef, periyot_uzunlugu_ay)
        
        # Sadece anlamlı sonuçları (geçiş olanları) listeye ekle
        if isinstance(sonuc, dict) and sonuc.get("Geçiş Yapan Müşteri Sayısı", 0) > 0:
            sonuc['Kaynak_Kategori'] = kaynak
            sonuc['Hedef_Kategori'] = hedef
            tum_sonuclar.append(sonuc)
            
    if not tum_sonuclar:
        return pd.DataFrame()
        
    sonuclar_df = pd.DataFrame(tum_sonuclar)
    
    # Daha anlamlı olması için kolon sırasını düzenle
    sonuclar_df = sonuclar_df[[
        'Kaynak_Kategori', 
        'Hedef_Kategori', 
        'Geçiş Yapan Müşteri Sayısı', 
        'Net Ciro Etkisi',
        'Kaybedilen Ciro (Kaynak Kategoriden)',
        'Kazanılan Ciro (Hedef Kategoriden)'
    ]]
    
    return sonuclar_df.sort_values('Geçiş Yapan Müşteri Sayısı', ascending=False)

def kategori_yasam_dongusu_analizi_yap(temiz_df):
    """
    GÜNCELLENMİŞ: Her bir ürün kategorisinin zaman içindeki aylık performansını
    birden fazla metrik (Ciro, Müşteri Sayısı, Kar Marjı) üzerinden hesaplar.
    """
    if 'Kategori' not in temiz_df.columns:
        return pd.DataFrame()

    # Her kategorinin, her aydaki performansını birden fazla metrikle hesapla
    aylik_kategori_performans = temiz_df.groupby([pd.Grouper(key='Tarih', freq='M'), 'Kategori']).agg(
        ToplamCiro=('ToplamTutar', 'sum'),
        BenzersizMusteriSayisi=('MusteriID', 'nunique'),
        NetKar=('NetKar', 'sum')
    ).reset_index()

    # Kar marjını hesapla (0'a bölme hatasını engelle)
    aylik_kategori_performans['KarMarji'] = aylik_kategori_performans.apply(
        lambda row: (row['NetKar'] / row['ToplamCiro']) * 100 if row['ToplamCiro'] > 0 else 0,
        axis=1
    )
    
    return aylik_kategori_performans.sort_values('Tarih')

def kategori_musteri_profili_analizi_yap(temiz_df, sonuclar_df):
    """
    Her bir ürün kategorisini satın alan müşteri kitlesinin segmentlere göre
    dağılımını (yüzdesel olarak) analiz eder.
    """
    if 'Kategori' not in temiz_df.columns:
        return pd.DataFrame()

    # Müşteri segmentlerini ham satış verileriyle birleştir
    # Bu sayede hangi segmentin hangi kategoriden alım yaptığını görebiliriz
    df_birlesik = pd.merge(
        temiz_df[['MusteriID', 'Kategori']],
        sonuclar_df[['Segment']],
        left_on='MusteriID',
        right_index=True,
        how='inner'
    ).drop_duplicates(subset=['MusteriID', 'Kategori']) # Her müşterinin bir kategoriden bir kez sayılmasını sağla

    # Her kategori için segment dağılımını (yüzdesel olarak) hesapla
    profil_df = pd.crosstab(df_birlesik['Kategori'], df_birlesik['Segment'], normalize='index') * 100
    
    return profil_df.sort_index()

def kategori_sepet_birlikteligi_yap(temiz_df, min_support=0.01):
    """
    Her bir alışveriş sepeti (aynı müşteri, aynı gün) içindeki kategorileri analiz ederek
    kategoriler arası birliktelik kurallarını bulur.
    """
    if 'Kategori' not in temiz_df.columns:
        return pd.DataFrame()

    # Sepetleri tanımla: Aynı müşteri, aynı gün yapılan alımlar tek bir sepettir.
    # drop_duplicates ile her sepette bir kategorinin sadece bir kez sayılmasını sağlıyoruz.
    sepetler = temiz_df[['MusteriID', 'Tarih', 'Kategori']].drop_duplicates()
    
    # Apriori algoritması için veriyi hazırla: one-hot encoding
    # Her satır bir sepet, her sütun bir kategori. Değerler 1 (sepette var) veya 0 (yok).
    sepet_matrisi = pd.crosstab(index=[sepetler['MusteriID'], sepetler['Tarih']], columns=sepetler['Kategori'])
    
    # Sıkça birlikte görülen kategori setlerini bul
    sik_kullanilan_kategoriler = apriori(sepet_matrisi, min_support=min_support, use_colnames=True)
    
    if sik_kullanilan_kategoriler.empty:
        return pd.DataFrame()
        
    # Birliktelik kurallarını oluştur
    kurallar = association_rules(sik_kullanilan_kategoriler, metric="lift", min_threshold=1)
    
    # Sonuçları daha anlamlı olacak şekilde filtreleyip sıralayalım
    kurallar = kurallar[kurallar['lift'] >= 1.1].sort_values('lift', ascending=False)
    
    return kurallar

def urun_performans_analizi_yap(temiz_df, baslangic_tarihi=None, bitis_tarihi=None):
    """
    Her bir ürünün performansını çeşitli metrikler üzerinden analiz eder.
    Eğer tarih aralığı belirtilirse, sadece o dönemdeki verilere göre hesaplama yapar.
    """
    df_analiz = temiz_df.copy()
    
    # Eğer başlangıç ve bitiş tarihleri verildiyse, ana veriyi filtrele
    if baslangic_tarihi and bitis_tarihi:
        df_analiz = df_analiz[
            (df_analiz['Tarih'].dt.date >= baslangic_tarihi) & 
            (df_analiz['Tarih'].dt.date <= bitis_tarihi)
        ]
        
    if 'UrunKodu' not in df_analiz.columns or df_analiz.empty:
        return pd.DataFrame()

    # Ürün bazında temel metrikleri hesapla
    urun_performans = df_analiz.groupby('UrunKodu').agg(
        Toplam_Ciro=('ToplamTutar', 'sum'),
        Toplam_Net_Kar=('NetKar', 'sum'),
        Benzersiz_Musteri_Sayisi=('MusteriID', 'nunique'),
        Toplam_Satis_Adedi=('Miktar', 'sum'),
        Islem_Sayisi=('MusteriID', 'count')
    ).reset_index()

    # Ek metrikleri hesapla
    urun_performans['Kar_Marji'] = (urun_performans['Toplam_Net_Kar'] / urun_performans['Toplam_Ciro']) * 100
    
    # NaN değerleri (0'a bölünme durumunda) temizle
    urun_performans.fillna(0, inplace=True)
    
    return urun_performans.sort_values('Toplam_Ciro', ascending=False)

def urun_icin_segment_profili(temiz_df, sonuclar_df, secilen_urun):
    """
    Belirli bir ürünü satın alan müşteri kitlesinin segmentlere göre
    dağılımını analiz eder.
    """
    # Seçilen ürünü alan tüm müşterilerin ID'lerini bul
    urunu_alan_musteriler = temiz_df[temiz_df['UrunKodu'] == secilen_urun]['MusteriID'].unique()
    
    # Bu müşterilerin segmentlerini ana sonuç tablosundan al
    profil_df = sonuclar_df[sonuclar_df.index.isin(urunu_alan_musteriler)]
    
    # Segment dağılımını say
    segment_dagilimi = profil_df['Segment'].value_counts()
    
    return segment_dagilimi

def segment_icin_urun_profili(temiz_df, sonuclar_df, secilen_segment):
    """
    Belirli bir müşteri segmentinin en çok satın aldığı ürünleri bulur.
    """
    # Seçilen segmente ait tüm müşteri ID'lerini bul
    segmente_ait_musteriler = sonuclar_df[sonuclar_df['Segment'] == secilen_segment].index
    
    # Bu müşterilerin yaptığı tüm alımları filtrele
    segment_alimlari = temiz_df[temiz_df['MusteriID'].isin(segmente_ait_musteriler)]
    
    # Bu segmentin en çok aldığı ürünleri ciroya göre sırala
    en_populer_urunler = segment_alimlari.groupby('UrunKodu')['ToplamTutar'].sum().nlargest(10)
    
    return en_populer_urunler

def create_ts_features(df):
    """
    Bir zaman serisi DataFrame'inden makine öğrenmesi için özellikler oluşturur.
    """
    df = df.copy()
    df['ay'] = df['ds'].dt.month
    df['yil'] = df['ds'].dt.year
    df['ceyrek'] = df['ds'].dt.quarter
    # Geçmiş satış verilerini özellik olarak ekle (lag features)
    for i in range(1, 7): # Son 6 ayın verisini özellik olarak kullan
        df[f'lag_{i}'] = df['y'].shift(i)
    df = df.dropna()
    return df

def random_forest_tahmin(df, tahmin_periyodu=6):
    """
    Zaman serisi verilerini özelliklerine ayırarak Random Forest Regressor ile tahmin yapar.
    """
    # 1. Özellik Mühendisliği
    df_ozellikli = create_ts_features(df)
    
    # Zaman serisi tahmininde, modelin eğitildiği son tarihi bilmek önemlidir.
    son_egitim_tarihi = df['ds'].iloc[-1]
    
    X = df_ozellikli.drop(['ds', 'y'], axis=1)
    y = df_ozellikli['y']
    
    # Yeterli veri yoksa (lag özellikleri oluşturulamadıysa)
    if X.empty:
        return None, None

    # 2. Model Eğitimi
    model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=2)
    model.fit(X, y)
    
    # 3. Adım Adım Gelecek Tahmini
    tahminler = []
    mevcut_veri = df['y'].tolist() # En son verilerle başla

    for i in range(tahmin_periyodu):
        # Tahmin yapılacak bir sonraki ay için özellikleri oluştur
        gelecek_tarih = son_egitim_tarihi + pd.DateOffset(months=i+1)
        
        yeni_ozellikler = {
            'ay': gelecek_tarih.month,
            'yil': gelecek_tarih.year,
            'ceyrek': gelecek_tarih.quarter,
            'lag_1': mevcut_veri[-1],
            'lag_2': mevcut_veri[-2],
            'lag_3': mevcut_veri[-3],
            'lag_4': mevcut_veri[-4],
            'lag_5': mevcut_veri[-5],
            'lag_6': mevcut_veri[-6],
        }
        
        # DataFrame'e çevir ve tahmin yap
        yeni_satir_df = pd.DataFrame([yeni_ozellikler], columns=X.columns)
        tahmin = model.predict(yeni_satir_df)[0]
        tahmin = max(0, tahmin) # Negatif satışı engelle
        
        # Yapılan tahmini bir sonraki adımda kullanılmak üzere listeye ekle
        tahminler.append(tahmin)
        mevcut_veri.append(tahmin)
        
    # Tahminleri standart DataFrame formatına getir
    gelecek_tarihler = pd.date_range(start=son_egitim_tarihi + pd.DateOffset(months=1), periods=tahmin_periyodu, freq='ME')
    tahmin_df = pd.DataFrame({'tahmin': tahminler}, index=gelecek_tarihler)
    
    return model, tahmin_df

def churn_tahmin_modeli_olustur(rfm_df, churn_limiti_gun=180):
    """
    GÜNCELLENMİŞ: RandomForestClassifier kullanarak daha doğru bir churn tahmin modeli oluşturur.
    SHAP analizi için gerekli olan X, X_train, modeli ve explainer'ı döndürür.
    """
    rfm_df['Churn'] = rfm_df['Recency'].apply(lambda x: 1 if x > churn_limiti_gun else 0)
    
    if len(rfm_df['Churn'].unique()) < 2:
        rfm_df['Churn_Olasiligi'] = 0
        return rfm_df, None, None, None, None, 0 # X_train için None eklendi

    X = rfm_df[['Recency', 'Frequency', 'Monetary']]
    y = rfm_df['Churn']
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5, class_weight='balanced')
    model.fit(X_train, y_train)
    
    dogruluk = accuracy_score(y_test, model.predict(X_test))
    rfm_df['Churn_Olasiligi'] = model.predict_proba(X)[:, 1]

    explainer = shap.TreeExplainer(model)
    
    # X_train'i de döndürerek global SHAP analizi için tutarlılık sağlıyoruz
    return rfm_df.sort_values('Churn_Olasiligi', ascending=False), model, explainer, X, X_train, dogruluk

def bireysel_churn_etkenlerini_hesapla(explainer, musteri_verisi_X):
    """
    NİHAİ VERSİYON: SHAP kütüphanesinin modern API'ını kullanarak, bir müşteriye ait
    tüm SHAP bilgilerini içeren tam bir Explanation nesnesi döndürür.
    """
    if explainer is None:
        return None
        
    # Tek bir müşteriye ait tüm SHAP bilgilerini (her iki sınıf için) bütün bir nesne olarak döndür.
    return explainer(musteri_verisi_X)

def anomali_nedenlerini_acikla(sonuclar_df):
    """
    Tespit edilen anormal müşterilerin, neden anormal olarak etiketlendiğini
    normal müşterilerin medyan değerleriyle kıyaslayarak basit kurallarla açıklar.
    """
    anormal_musteriler = sonuclar_df[sonuclar_df['Anomali_Etiketi'] == -1]
    normal_musteriler = sonuclar_df[sonuclar_df['Anomali_Etiketi'] == 1]
    
    if normal_musteriler.empty or anormal_musteriler.empty:
        return {}

    # Normal müşterilerin profilini (medyan) çıkar
    medyanlar = normal_musteriler[['Recency', 'Frequency', 'Monetary']].median()
    
    anomali_nedenleri = {}
    for index, musteri in anormal_musteriler.iterrows():
        nedenler = []
        # Kuralları uygula
        if musteri['Monetary'] > medyanlar['Monetary'] * 5:
            nedenler.append("Aşırı Yüksek Harcama")
        if musteri['Frequency'] > medyanlar['Frequency'] * 5:
            nedenler.append("Aşırı Sık Alışveriş")
        if musteri['Recency'] > medyanlar['Recency'] * 3 and musteri['Monetary'] > medyanlar['Monetary']:
             nedenler.append("Eski ve Değerli Müşteri (Pasifleşme Riski)")
        elif musteri['Recency'] > medyanlar['Recency'] * 4:
            nedenler.append("Çok Uzun Süredir Pasif")
        
        if musteri['Frequency'] == 1 and musteri['Monetary'] > medyanlar['Monetary'] * 5:
             nedenler.append("Tek Seferlik Dev Alıcı (Whale)")

        # Eğer hiçbir kurala uymuyorsa genel bir etiket ver
        if not nedenler:
            nedenler.append("Genel Davranış Dışında")
            
        anomali_nedenleri[index] = ", ".join(nedenler)
        
    return anomali_nedenleri

def segmente_benzer_musteri_bul(sonuclar_df, kaynak_segment, top_n=10):
    """
    Bir kaynak segmentin ortalama RFM profiline en çok benzeyen, ancak o segmentte olmayan
    diğer müşterileri bulur.
    """
    if kaynak_segment not in sonuclar_df['Segment'].unique():
        return pd.DataFrame()

    # Benzerlik için kullanılacak RFM özelliklerini ölçeklendir
    rfm_ozellikler = sonuclar_df[['Recency', 'Frequency', 'Monetary']]
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_ozellikler)
    rfm_scaled_df = pd.DataFrame(rfm_scaled, index=sonuclar_df.index, columns=rfm_ozellikler.columns)
    
    # Kaynak segmenti ve diğer müşterileri ayır
    kaynak_musteri_idler = sonuclar_df[sonuclar_df['Segment'] == kaynak_segment].index
    hedef_musteri_idler = sonuclar_df[sonuclar_df['Segment'] != kaynak_segment].index
    
    kaynak_df_scaled = rfm_scaled_df.loc[kaynak_musteri_idler]
    hedef_df_scaled = rfm_scaled_df.loc[hedef_musteri_idler]

    if kaynak_df_scaled.empty or hedef_df_scaled.empty:
        return pd.DataFrame()

    # Kaynak segmentin ortalama profilini (centroid) hesapla
    segment_centroid = kaynak_df_scaled.mean().values.reshape(1, -1)
    
    # Hedef müşterilerin bu profile olan benzerliğini hesapla
    benzerlik_skorlari = cosine_similarity(hedef_df_scaled.values, segment_centroid)
    
    # Sonuçları bir DataFrame'e dönüştür
    sonuc_df = pd.DataFrame(benzerlik_skorlari, index=hedef_df_scaled.index, columns=['Benzerlik_Skoru'])
    sonuc_df = sonuc_df.sort_values('Benzerlik_Skoru', ascending=False)
    
    # En çok benzeyen ilk N müşteriyi al ve orijinal verileriyle birleştir
    en_benzerler = sonuc_df.head(top_n)
    en_benzerler_detayli = en_benzerler.merge(sonuclar_df, left_index=True, right_index=True)
    
    return en_benzerler_detayli.sort_values('Benzerlik_Skoru', ascending=False)

@st.cache_data
def _musteri_urun_setlerini_getir(temiz_df):
    
    """
    (Yardımcı Fonksiyon) Her müşterinin satın aldığı benzersiz ürünlerin setini oluşturur.
    Tekrar tekrar hesaplanmaması için cache'lenir.
    """
    return temiz_df.groupby('MusteriID')['UrunKodu'].apply(set)

def urun_benzerligi_hesapla(temiz_df, kaynak_musteri_id):
    """
    Bir kaynak müşterinin ürün zevkinin diğer tüm müşterilere olan 
    Jaccard benzerliğini hesaplar.
    """
    musteri_urun_setleri = _musteri_urun_setlerini_getir(temiz_df)
    
    if kaynak_musteri_id not in musteri_urun_setleri:
        return pd.Series(dtype='float64')

    kaynak_set = musteri_urun_setleri[kaynak_musteri_id]
    benzerlik_skorlari = {}

    for musteri_id, hedef_set in musteri_urun_setleri.items():
        if musteri_id == kaynak_musteri_id:
            continue
        
        intersection_size = len(kaynak_set.intersection(hedef_set))
        union_size = len(kaynak_set.union(hedef_set))
        
        if union_size == 0:
            benzerlik_skorlari[musteri_id] = 0
        else:
            benzerlik_skorlari[musteri_id] = intersection_size / union_size
            
    return pd.Series(benzerlik_skorlari)

def kume_profillerini_etiketle(merkezler_df, genel_ortalamalar):
    """
    Oluşturulan küme merkezlerini (centroid), veri setinin genel ortalamalarıyla
    kıyaslayarak her kümeye otomatik, açıklayıcı bir etiket atar.
    """
    yeni_isimler = {}
    for index, satir in merkezler_df.iterrows():
        ozellikler = []
        
        # Recency Değerlendirmesi (Düşük olması iyi)
        if satir['Recency'] < genel_ortalamalar['Recency'] * 0.75:
            ozellikler.append("Aktif")
        elif satir['Recency'] > genel_ortalamalar['Recency'] * 1.5:
            ozellikler.append("Pasifleşmiş")

        # Frequency Değerlendirmesi (Yüksek olması iyi)
        if satir['Frequency'] > genel_ortalamalar['Frequency'] * 1.5:
            ozellikler.append("Sık Alışveriş Yapan")
        elif satir['Frequency'] < genel_ortalamalar['Frequency'] * 0.75:
            ozellikler.append("Seyrek Alışveriş Yapan")
            
        # Monetary Değerlendirmesi (Yüksek olması iyi)
        if satir['Monetary'] > genel_ortalamalar['Monetary'] * 1.5:
            ozellikler.append("Yüksek Harcayan")
        elif satir['Monetary'] < genel_ortalamalar['Monetary'] * 0.75:
            ozellikler.append("Düşük Harcayan")

        if not ozellikler:
            final_etiket = f"Ortalama Profil - Küme {index}"
        else:
            final_etiket = " & ".join(ozellikler)
            
        yeni_isimler[index] = final_etiket
        
    return yeni_isimler

# Bu sözlük, her bir özelliğin "anlamını" tanımlar.
FEATURE_METADATA = {
    'Recency': {'good': 'Aktif', 'bad': 'Pasifleşmiş', 'direction': 'low'},
    'Frequency': {'good': 'Sık Alışveriş Yapan', 'bad': 'Seyrek Alışveriş Yapan', 'direction': 'high'},
    'Monetary': {'good': 'Yüksek Harcayan', 'bad': 'Düşük Harcayan', 'direction': 'high'},
    'CLV_Net_Kar': {'good': 'Yüksek Değerli', 'bad': 'Düşük Değerli', 'direction': 'high'},
    'Churn_Olasiligi': {'good': 'Sadık', 'bad': 'Yüksek Riskli', 'direction': 'low'},
    'MPS': {'good': 'Yüksek Performanslı', 'bad': 'Düşük Performanslı', 'direction': 'high'}
}

def dinamik_kume_etiketle(merkezler_df, genel_ortalamalar):
    """
    YENİ ve AKILLI FONKSİYON: Seçilen herhangi bir özellik kombinasyonuna göre,
    FEATURE_METADATA sözlüğünü kullanarak dinamik olarak açıklayıcı etiketler atar.
    """
    yeni_isimler = {}
    secilen_ozellikler = merkezler_df.columns.tolist()

    for index, satir in merkezler_df.iterrows():
        ozellikler = []
        
        for ozellik in secilen_ozellikler:
            if ozellik in FEATURE_METADATA:
                meta = FEATURE_METADATA[ozellik]
                deger = satir[ozellik]
                ortalama = genel_ortalamalar[ozellik]

                if meta['direction'] == 'high': # Yüksek değerin iyi olduğu metrikler
                    if deger > ortalama * 1.25:
                        ozellikler.append(meta['good'])
                    elif deger < ortalama * 0.75:
                        ozellikler.append(meta['bad'])
                
                elif meta['direction'] == 'low': # Düşük değerin iyi olduğu metrikler
                    if deger < ortalama * 0.75:
                        ozellikler.append(meta['good'])
                    elif deger > ortalama * 1.25:
                        ozellikler.append(meta['bad'])

        if not ozellikler:
            final_etiket = f"Ortalama Profil - Küme {index}"
        else:
            final_etiket = " & ".join(ozellikler)
            
        yeni_isimler[index] = final_etiket
        
    return yeni_isimler

def deger_gocu_analizi_yap(temiz_df, sonuclar_df, baslangic1, bitis1, baslangic2, bitis2):
    """
    İki zaman periyodu arasında müşterilerin CLV'ye dayalı değer segmentleri
    arasındaki geçişini (göçünü) analiz eder.
    """
    # 1. Her periyotta aktif olan müşterileri bul
    musteriler_p1 = set(temiz_df[
        (temiz_df['Tarih'].dt.date >= baslangic1) & (temiz_df['Tarih'].dt.date <= bitis1)
    ]['MusteriID'].unique())
    
    musteriler_p2 = set(temiz_df[
        (temiz_df['Tarih'].dt.date >= baslangic2) & (temiz_df['Tarih'].dt.date <= bitis2)
    ]['MusteriID'].unique())

    # 2. Değer segmenti tr eşiklerini tüm müşteri tabanının CLV'sine göre belirle
    try:
        # qcut 3'e bölmek için 4 sınır noktasına ihtiyaç duyar [0, 0.33, 0.66, 1]
        sonuclar_df['Deger_Segmenti'] = pd.qcut(sonuclar_df['CLV_Net_Kar'], q=3, labels=["Düşük Değerli", "Orta Değerli", "Yüksek Değerli"])
    except ValueError: # Yeterli çeşitlilik yoksa
        sonuclar_df['Deger_Segmenti'] = "Tek Değer Grubu"
        
    # 3. Müşteri göçünü hesapla
    goc_verisi = []
    tum_musteriler = musteriler_p1.union(musteriler_p2)

    for musteri_id in tum_musteriler:
        if musteri_id not in sonuclar_df.index: continue # Ana tabloda olmayan bir ID ise atla

        deger_segmenti = sonuclar_df.loc[musteri_id, 'Deger_Segmenti']
        
        # Periyot 1'deki durum
        if musteri_id in musteriler_p1:
            segment_p1 = deger_segmenti
        else:
            segment_p1 = "Yeni Müşteri"
            
        # Periyot 2'deki durum
        if musteri_id in musteriler_p2:
            segment_p2 = deger_segmenti
        else:
            segment_p2 = "Pasif / Churn"
            
        goc_verisi.append({
            'Onceki_Durum': segment_p1,
            'Simdiki_Durum': segment_p2
        })
        
    if not goc_verisi:
        return pd.DataFrame()

    goc_df = pd.DataFrame(goc_verisi)
    
    # Sankey için veriyi grupla
    sankey_data = goc_df.groupby(['Onceki_Durum', 'Simdiki_Durum']).size().reset_index(name='deger')
    return sankey_data

def trend_analizi_yap(trend_serisi):
    """
    seasonal_decompose'dan gelen trend bileşenini analiz eder.
    Trendin yönünü ve eğimini (ortalama aylık değişim) hesaplar.
    """
    trend_temiz = trend_serisi.dropna()
    
    if len(trend_temiz) < 2:
        return None

    # Trendin yönünü belirle
    yon = "Yükselişte" if trend_temiz.iloc[-1] > trend_temiz.iloc[0] else "Düşüşte"
    
    # Trendin eğimini (slope) hesapla - lineer regresyon
    y = trend_temiz.values
    x = np.arange(len(y))
    egim, _ = np.polyfit(x, y, 1)
    
    return {'yon': yon, 'egim': egim}

def mevsimsellik_analizi_yap(mevsimsellik_serisi):
    """
    seasonal_decompose'dan gelen mevsimsellik bileşenini analiz eder.
    Her ayın ortalama mevsimsel etkisini bar grafiği için hazırlar.
    """
    if mevsimsellik_serisi.dropna().empty:
        return pd.DataFrame()

    df = pd.DataFrame({'etki': mevsimsellik_serisi})
    df['ay_no'] = df.index.month
    
    aylik_etki = df.groupby('ay_no')['etki'].mean().reset_index()
    
    ay_isimleri = {
        1: 'Ocak', 2: 'Şubat', 3: 'Mart', 4: 'Nisan', 5: 'Mayıs', 6: 'Haziran',
        7: 'Temmuz', 8: 'Ağustos', 9: 'Eylül', 10: 'Ekim', 11: 'Kasım', 12: 'Aralık'
    }
    aylik_etki['Ay'] = aylik_etki['ay_no'].map(ay_isimleri)
    
    return aylik_etki[['Ay', 'etki', 'ay_no']].sort_values('ay_no')

def sonraki_kategori_onerisi(migrasyon_matrisi, kaynak_kategori):
    """
    Bir başlangıç kategorisinden yola çıkarak, müşterilere tanıtılması en
    olası olan sonraki kategorileri, geçiş oranına göre sıralar.
    """
    if kaynak_kategori not in migrasyon_matrisi.index:
        return pd.Series(dtype='float64') # Boş bir seri döndür

    # Kaynak kategoriye ait geçiş olasılıklarını al ve en yüksekten aza doğru sırala
    oneriler = migrasyon_matrisi.loc[kaynak_kategori].sort_values(ascending=False)
    
    # Kendisine geçişi (eğer varsa) ve %0 olanları listeden çıkar
    oneriler = oneriler[oneriler > 0]
    if kaynak_kategori in oneriler.index:
        oneriler = oneriler.drop(kaynak_kategori)
        
    return oneriler

