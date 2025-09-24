# 0_Genel_Bakış.py
# SORUMLULUĞU: Ana sayfa. Uygulamanın başlangıç noktası ve tüm veriyi işleyen ana fonksiyonun sahibi.

import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
from data_handler import veriyi_yukle_ve_temizle, genel_satis_trendi_hazirla
from analysis_engine import (rfm_skorlarini_hesapla, 
                           musterileri_segmentle, 
                           churn_tahmin_modeli_olustur, 
                           clv_hesapla,
                           market_basket_analizi_yap,
                           genel_rapor_pdf_olustur,
                           anomali_tespiti_yap,
                           davranissal_anomali_tespiti_yap)

st.set_page_config(page_title="Genel Bakış", layout="wide")

# --- 1. VERİ YÜKLEME VE TÜM ANALİZLER ---
@st.cache_data
def veriyi_getir_ve_isle():
    dosya_adi = 'satis_verileri_guncellenmis.json' 
    temiz_df = veriyi_yukle_ve_temizle(dosya_adi)
    rfm_df = rfm_skorlarini_hesapla(temiz_df)
    segmentli_df = musterileri_segmentle(rfm_df)
    churn_df, _, _, _, _, _ = churn_tahmin_modeli_olustur(segmentli_df)
    clv_df = clv_hesapla(churn_df)
    sonuclar_df = clv_df.copy()
    if 'MusteriAdi' not in sonuclar_df.columns:
        sonuclar_df['MusteriAdi'] = sonuclar_df.index
    # Anomali listelerini de burada oluşturup session_state'e atalım
    profil_anomalileri_df = anomali_tespiti_yap(sonuclar_df.copy())
    st.session_state.profil_anomalileri = profil_anomalileri_df[profil_anomalileri_df['Anomali_Etiketi'] == -1].index.tolist()
    davranissal_anomaliler_df = davranissal_anomali_tespiti_yap(temiz_df)
    st.session_state.davranissal_anomaliler = davranissal_anomaliler_df['MusteriID'].tolist()
    return temiz_df, sonuclar_df

temiz_df, sonuclar_df = veriyi_getir_ve_isle()

st.title("📊 Müşteri Analitiği Genel Bakış Panosu")

# --- 2. KONTROL PANELİ (FİLTRELER VE İNDİRME) ---
# Tüm kontrolleri açılıp kapanabilir bir expander içine koyuyoruz.
with st.expander("⚙️ Kontrol Paneli: Filtreler ve İndirme Seçenekleri", expanded=True):
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("##### Tarih Aralığı Seçimi")
        min_tarih = temiz_df['Tarih'].min()
        max_tarih = temiz_df['Tarih'].max()
        secilen_baslangic_tarihi = st.date_input("Başlangıç Tarihi", min_tarih, min_value=min_tarih, max_value=max_tarih)
        secilen_bitis_tarihi = st.date_input("Bitiş Tarihi", max_tarih, min_value=min_tarih, max_value=max_tarih)
        
    with col2:
        st.markdown("##### Segment Seçimi")
        if isinstance(sonuclar_df['Segment'].dtype, pd.CategoricalDtype):
            segment_listesi = ['Tümü'] + sonuclar_df['Segment'].cat.categories.tolist()
        else:
            segment_listesi = ['Tümü'] + sonuclar_df['Segment'].unique().tolist()
        secilen_segmentler = st.multiselect("Müşteri Segmentlerini Seçin", segment_listesi, default=['Tümü'])

    # İndirme butonları için veriyi hazırlama fonksiyonu
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=True).encode('utf-8-sig')

    # Filtreleme mantığını burada çalıştırıyoruz
    aktif_musteriler = temiz_df[(temiz_df['Tarih'].dt.date >= secilen_baslangic_tarihi) & (temiz_df['Tarih'].dt.date <= secilen_bitis_tarihi)]['MusteriID'].unique()
    tarih_filtrelenmis_df = sonuclar_df[sonuclar_df.index.isin(aktif_musteriler)]
    if 'Tümü' in secilen_segmentler or not secilen_segmentler:
        nihai_filtrelenmis_df = tarih_filtrelenmis_df
    else:
        nihai_filtrelenmis_df = tarih_filtrelenmis_df[tarih_filtrelenmis_df['Segment'].isin(secilen_segmentler)]

    with col3:
        st.markdown("##### İndirme Seçenekleri")
        # CSV İndirme
        csv = convert_df_to_csv(nihai_filtrelenmis_df)
        st.download_button(
            label="⬇️ Veriyi İndir (.csv)",
            data=csv,
            file_name='musteri_analitik_sonuclari.csv',
            mime='text/csv',
            use_container_width=True
        )

# --- 4. ANA SAYFA İÇERİĞİ ---
st.title("📊 Genel Bakış ve KPI'lar")
st.markdown("---")

st.header("Genel Satış Trendi")
aylik_satislar = genel_satis_trendi_hazirla(temiz_df)
fig_trend = px.area(aylik_satislar, x='ds', y='y', title='Aylık Toplam Satış Cirosu (Tüm Zamanlar)', labels={'ds': 'Tarih', 'y': 'Toplam Ciro (€)'},
                    color_discrete_sequence=['#6c63ff'])
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")
st.header("Filtrelenmiş Segment Özeti")
st.markdown(f"**`{secilen_baslangic_tarihi.strftime('%d-%m-%Y')}`** ve **`{secilen_bitis_tarihi.strftime('%d-%m-%Y')}`** tarihleri arasında aktif olan müşterilerin analizi")

# Dönemsel KPI Hesaplaması
donem_df = temiz_df[
    (temiz_df['Tarih'].dt.date >= secilen_baslangic_tarihi) & 
    (temiz_df['Tarih'].dt.date <= secilen_bitis_tarihi)
]
donem_df_filtrelenmis = donem_df[donem_df['MusteriID'].isin(nihai_filtrelenmis_df.index)]
donem_cirosu = donem_df_filtrelenmis['ToplamTutar'].sum()
donem_net_kari = donem_df_filtrelenmis['NetKar'].sum() # Net Kar hesaplaması eklendi
donem_islem_sayisi = len(donem_df_filtrelenmis)

profil_anomali_sayisi = len(set(st.session_state.get('profil_anomalileri', [])).intersection(set(nihai_filtrelenmis_df.index)))
davranissal_anomali_sayisi = len(set(st.session_state.get('davranissal_anomaliler', [])).intersection(set(nihai_filtrelenmis_df.index)))
toplam_anomali = profil_anomali_sayisi + davranissal_anomali_sayisi

# KPI Kartları
col1, col2, col3 = st.columns(3)
col1.metric("Aktif Müşteri Sayısı", f"{nihai_filtrelenmis_df.shape[0]:,}")
col2.metric("Seçilen Dönemdeki Ciro", f"{donem_cirosu:,.0f} €")
col3.metric("Seçilen Dönemdeki Net Kar", f"{donem_net_kari:,.0f} €") # Yeni KPI

col4, col5, col6 = st.columns(3)
col4.metric("Seçilen Dönemdeki İşlem Sayısı", f"{donem_islem_sayisi:,}")
col5.metric("Ortalama Yaşam Boyu Değeri (CLV)", f"{nihai_filtrelenmis_df['CLV_Net_Kar'].mean():,.0f} €")
col6.metric("⚠️ Anormal Müşteri Sayısı", f"{toplam_anomali}", help="Anomali Tespiti sayfasında analizi çalıştırdıktan sonra burada görünür.")
            

st.markdown("---")

col1_chart, col2_chart = st.columns(2)
fig_pie, fig_bar = None, None
with col1_chart:
    st.subheader("Müşteri Segment Dağılımı")
    segment_dagilimi = nihai_filtrelenmis_df['Segment'].value_counts()
    if not segment_dagilimi.empty:
        fig_pie = px.pie(values=segment_dagilimi.values, names=segment_dagilimi.index, title="Filtrelenmiş Müşterilerin Segment Dağılımı",
                         color_discrete_sequence=px.colors.sequential.Plasma_r)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Bu filtrede gösterilecek müşteri bulunmuyor.")

with col2_chart:
    st.subheader("Segment Bazında Ortalama CLV")
    if not nihai_filtrelenmis_df.empty:
        segment_clv = nihai_filtrelenmis_df.groupby('Segment')['CLV_Net_Kar'].mean().sort_values(ascending=False)
        if not segment_clv.empty:
            fig_bar = px.bar(segment_clv, x=segment_clv.index, y='CLV_Net_Kar', title="Segmentlerin Ortalama Değeri",
                             labels={'CLV_Net_Kar': 'Ortalama CLV (€)', 'Segment': 'Müşteri Segmenti'},
                             color=segment_clv.index, color_discrete_sequence=px.colors.sequential.Plasma_r)
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Bu filtrede gösterilecek CLV verisi bulunmuyor.")

st.markdown("---")
st.header("Segment Detayları (Filtrelenmiş Veri)")
st.markdown("Aşağıdaki sekmelerden birini seçerek, seçtiğiniz filtrelere uyan müşteri listelerini detaylı olarak inceleyebilirsiniz.")

profil_anomalileri = st.session_state.get('profil_anomalileri', [])
davranissal_anomaliler = st.session_state.get('davranissal_anomaliler', [])

if isinstance(sonuclar_df['Segment'].dtype, pd.CategoricalDtype):
    segment_tab_listesi = sonuclar_df['Segment'].cat.categories.tolist()
    segment_tabs = st.tabs(segment_tab_listesi)

    for i, segment_adi in enumerate(segment_tab_listesi):
        with segment_tabs[i]:
            segment_df = nihai_filtrelenmis_df[nihai_filtrelenmis_df['Segment'] == segment_adi]
            
            if segment_df.empty:
                st.info("Seçtiğiniz filtrelere uyan bu segmentte müşteri bulunmamaktadır.")
            else:
                # Anomali ikonlarını ekle
                segment_df['MusteriAdi_Display'] = segment_df['MusteriAdi'].apply(
                    lambda x: f"⚠️ {x}" if (x in profil_anomalileri or x in davranissal_anomaliler) else x
                )
                
                st.metric(f"Müşteri Sayısı", segment_df.shape[0])
                st.dataframe(segment_df[['MusteriAdi_Display', 'MPS', 'CLV_Net_Kar', 'Churn_Olasiligi', 'Recency', 'Frequency', 'Monetary']]
                             .rename(columns={'MusteriAdi_Display': 'Müşteri Adı'})
                             .sort_values('CLV_Net_Kar', ascending=False)
                             .style.format({
                                 'MPS': '{:.0f}', 'CLV_Net_Kar': '{:,.0f} €', 'Churn_Olasiligi': '{:.1%}',
                                 'Recency': '{} gün', 'Frequency': '{} adet', 'Monetary': '{:,.0f} €'
                             }))
else:
    st.warning("Segment verisi bulunamadı.")