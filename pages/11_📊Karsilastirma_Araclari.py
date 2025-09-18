# pages/11_Karşılaştırma_Araçları.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
# Gerekli fonksiyonları merkezi modüllerden import edelim
from data_handler import veriyi_yukle_ve_temizle
from analysis_engine import (rfm_skorlarini_hesapla, musterileri_segmentle, 
                           churn_tahmin_modeli_olustur, clv_hesapla,
                           donemsel_analiz_yap, benchmark_profili_hesapla, deger_gocu_analizi_yap)

st.set_page_config(page_title="Karşılaştırma Araçları", layout="wide")

@st.cache_data
def veriyi_getir_ve_isle():
    dosya_adi = 'satis_verileri.json' 
    temiz_df = veriyi_yukle_ve_temizle(dosya_adi)
    rfm_df = rfm_skorlarini_hesapla(temiz_df)
    segmentli_df = musterileri_segmentle(rfm_df)
    churn_df, _, _, _, _, _ = churn_tahmin_modeli_olustur(segmentli_df)
    clv_df = clv_hesapla(churn_df)
    sonuclar_df = clv_df.copy()
    if 'MusteriAdi' not in sonuclar_df.columns:
        sonuclar_df['MusteriAdi'] = sonuclar_df.index
    return temiz_df, sonuclar_df

temiz_df, sonuclar_df = veriyi_getir_ve_isle()

st.title("📊 Karşılaştırma Araçları")
st.markdown("Bu modül, müşterileri, segmentleri ve farklı zaman periyotlarını çeşitli metrikler üzerinden birbirleriyle ve benchmark profilleriyle karşılaştırmanızı sağlar.")



tab1, tab2, tab3 = st.tabs(["👥 Müşteri Karşılaştırma", "📈 Segment Performansı", "🗓️ Dönemsel Karşılaştırma"])

# --- SEKME 1: MÜŞTERİ KARŞILAŞTIRMA (Benchmark ile) ---
with tab1:
    st.header("Müşteri Profillerini Yan Yana Karşılaştır")
    
    col1, col2 = st.columns(2)
    with col1:
        musteri_listesi = sonuclar_df['MusteriAdi'].tolist()
        secilen_musteriler = st.multiselect(
            "Karşılaştırmak için 2 veya daha fazla müşteri seçin:",
            options=musteri_listesi,
            max_selections=4 # Benchmark için yer ayır
        )
    with col2:
        benchmark_secenekleri = ['Tüm Müşteriler'] + sonuclar_df['Segment'].unique().tolist()
        secilen_benchmark = st.selectbox(
            "Hangi profile göre kıyaslama yapılsın?",
            options=benchmark_secenekleri
        )

    if len(secilen_musteriler) >= 1:
        karsilastirma_df = sonuclar_df[sonuclar_df['MusteriAdi'].isin(secilen_musteriler)].copy()
        
        benchmark_profili = benchmark_profili_hesapla(sonuclar_df, secilen_benchmark)
        # Benchmark profilini DataFrame'e eklerken sütunların eşleştiğinden emin olalım
        benchmark_df_row = benchmark_profili.to_frame().T
        benchmark_df_row['MusteriAdi'] = benchmark_profili.name
        karsilastirma_df = pd.concat([karsilastirma_df, benchmark_df_row], ignore_index=True)
        
        st.subheader("Metrik Karşılaştırma Tablosu")
        
        # --- DÜZELTİLMİŞ BÖLÜM ---
        # Her satır için özel bir format tanımlayan bir sözlük oluşturuyoruz
        format_sozlugu = {
            'Recency': '{:.0f}',
            'Frequency': '{:.0f}',
            'Monetary': '{:,.0f} €',
            'MPS': '{:.0f}',
            'CLV_Net_Kar': '{:,.0f} €',
            'Churn_Olasiligi': '{:.1%}',
            'R_Score': '{:.0f}',
            'F_Score': '{:.0f}',
            'M_Score': '{:.0f}',
            # 'Segment' ve 'MusteriAdi' gibi metin satırları için bir format belirtmiyoruz
        }
        
        # Tabloyu bu sözlükle formatlıyoruz
        st.dataframe(karsilastirma_df[['MusteriAdi', 'Segment', 'Recency', 'Frequency', 'Monetary', 'MPS', 'CLV_Net_Kar', 'Churn_Olasiligi']]
                     .set_index('MusteriAdi').T
                     .style.format(formatter=format_sozlugu, na_rep="-"))
        # --- DÜZELTME SONU ---
        
        st.subheader("RFM Profili Karşılaştırması (Radar Grafiği)")
        rfm_data = karsilastirma_df[['R_Score', 'F_Score', 'M_Score']].fillna(0)
        scaler = MinMaxScaler()
        rfm_scaled = scaler.fit_transform(rfm_data)
        
        fig = go.Figure()
        categories = ['Recency Skoru', 'Frequency Skoru', 'Monetary Skoru']
        
        for i, musteri_adi in enumerate(karsilastirma_df['MusteriAdi']):
            fig.add_trace(go.Scatterpolar(r=rfm_scaled[i], theta=categories, fill='toself', name=musteri_adi))
            
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

# --- SEKME 2 ve 3 (Değişiklik yok) ---
with tab2:
    st.header("Segment Performanslarını Karşılaştır")
    
    segment_listesi_tab2 = sonuclar_df['Segment'].unique().tolist()
    secilen_segmentler_tab2 = st.multiselect(
        "Karşılaştırmak için 2 veya daha fazla segment seçin:",
        options=segment_listesi_tab2,
        default=segment_listesi_tab2[:min(len(segment_listesi_tab2), 2)] # Varsayılan olarak ilk 2 segmenti seç
    )

    if len(secilen_segmentler_tab2) >= 1:
        # Karşılaştırma için seçilen segmentlerdeki tüm müşterileri filtrele
        segment_verisi_filtrelenmis = sonuclar_df[sonuclar_df['Segment'].isin(secilen_segmentler_tab2)]

        # Ortalama metrikler tablosu
        segment_ortalamalari = segment_verisi_filtrelenmis.groupby('Segment').agg({
            'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean',
            'CLV_Net_Kar': 'mean', 'Churn_Olasiligi': 'mean'
        })
        st.subheader("Segmentlerin Ortalama Metrikleri")
        st.dataframe(segment_ortalamalari.T.style.format("{:,.1f}"))
        
        # Ortalama CLV ve Churn karşılaştırma grafiği
        st.subheader("Metriklerin Görsel Karşılaştırması (Genel Ortalamaya Göre)")
        benchmark = sonuclar_df[['CLV_Net_Kar', 'Churn_Olasiligi']].mean()
        plot_df_melted = segment_ortalamalari[['CLV_Net_Kar', 'Churn_Olasiligi']].reset_index().melt(id_vars='Segment', var_name='Metrik', value_name='Değer')
        fig_bar = px.bar(plot_df_melted, x='Metrik', y='Değer', color='Segment', 
                         barmode='group', title="Segmentlerin Ortalama CLV ve Churn Olasılığı")
        fig_bar.add_hline(y=benchmark['CLV_Net_Kar'], line_dash="dash", line_color="blue", annotation_text="Ortalama CLV")
        fig_bar.add_hline(y=benchmark['Churn_Olasiligi'], line_dash="dash", line_color="red", annotation_text="Ortalama Churn")
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- YENİ EKLENEN BÖLÜM: Kutu Grafikleri ile Dağılım Analizi ---
        st.markdown("---")
        st.subheader("Segment Metrik Dağılımları (Kutu Grafiği)")
        st.markdown("Bu grafikler, her segmentin içindeki müşteri dağılımının ne kadar homojen veya değişken olduğunu gösterir.")
        
        secilen_metrik_kutu = st.selectbox(
            "Dağılımını görmek istediğiniz metriği seçin:",
            ('Monetary', 'Recency', 'Frequency', 'CLV_Net_Kar')
        )
        
        if secilen_metrik_kutu:
            fig_box = px.box(
                segment_verisi_filtrelenmis,
                x='Segment',
                y=secilen_metrik_kutu,
                color='Segment',
                title=f"'{secilen_metrik_kutu}' Metriğinin Segmentlere Göre Dağılımı",
                points="outliers" # Aykırı değerleri göster
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            with st.expander("📊 Kutu Grafiği Nasıl Yorumlanır?"):
                st.info("""
                - **Kutunun Ortasındaki Çizgi:** Medyan (ortalama değil, tam orta değerdeki müşteri) değerini gösterir.
                - **Kutunun Alt ve Üst Kenarları:** Müşterilerin %50'sinin bulunduğu aralığı (25. ve 75. yüzdelikler) gösterir. Kutu ne kadar kısaysa, segment o kadar homojendir.
                - **Çizgiler (Whiskers):** Verinin genel yayılımını gösterir.
                - **Noktalar:** Segmentin genelinden çok farklı olan aykırı (outlier) müşterileri temsil eder.
                """)

with tab3:
    st.header("İki Farklı Zaman Periyodunun Performansını Karşılaştır")
    
    st.markdown("---")
    st.markdown("**Hızlı Filtreler**")
    bugun = date.today()
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("Bu Ay vs Geçen Ay", use_container_width=True):
            st.session_state.p2_start = bugun.replace(day=1)
            st.session_state.p2_end = bugun
            gecen_ay_sonu = st.session_state.p2_start - timedelta(days=1)
            st.session_state.p1_start = gecen_ay_sonu.replace(day=1)
            st.session_state.p1_end = gecen_ay_sonu
            
    with col_btn2:
        if st.button("Son 30 Gün vs Önceki 30 Gün", use_container_width=True):
            st.session_state.p2_end = bugun
            st.session_state.p2_start = bugun - timedelta(days=29)
            st.session_state.p1_end = st.session_state.p2_start - timedelta(days=1)
            st.session_state.p1_start = st.session_state.p1_end - timedelta(days=29)

    with col_btn3:
        if st.button("Bu Yıl vs Geçen Yıl", use_container_width=True):
            st.session_state.p2_start = bugun.replace(month=1, day=1)
            st.session_state.p2_end = bugun
            st.session_state.p1_start = st.session_state.p2_start - relativedelta(years=1)
            st.session_state.p1_end = bugun - relativedelta(years=1)
    st.markdown("---")
    
    bu_ay_basi = bugun.replace(day=1)
    gecen_ay_sonu = bu_ay_basi - timedelta(days=1)
    gecen_ay_basi = gecen_ay_sonu.replace(day=1)
    
    col1_tarih, col2_tarih = st.columns(2)
    with col1_tarih:
        st.markdown("**Periyot 1**")
        baslangic1 = st.date_input("Başlangıç Tarihi", value=st.session_state.get("p1_start", gecen_ay_basi), key="d1_start")
        bitis1 = st.date_input("Bitiş Tarihi", value=st.session_state.get("p1_end", gecen_ay_sonu), key="d1_end")
    with col2_tarih:
        st.markdown("**Periyot 2**")
        baslangic2 = st.date_input("Başlangıç Tarihi", value=st.session_state.get("p2_start", bu_ay_basi), key="d2_start")
        bitis2 = st.date_input("Bitiş Tarihi", value=st.session_state.get("p2_end", bugun), key="d2_end")
        
    if st.button("Dönemleri Karşılaştır", type="primary"):
        with st.spinner("İki dönem için metrikler hesaplanıyor..."):
            donemsel_sonuclar = donemsel_analiz_yap(temiz_df, baslangic1, bitis1, baslangic2, bitis2)
            deger_gocu_verisi = deger_gocu_analizi_yap(temiz_df, sonuclar_df, baslangic1, bitis1, baslangic2, bitis2)
        
        st.session_state.donemsel_sonuclar = donemsel_sonuclar
        st.session_state.deger_gocu_verisi = deger_gocu_verisi
        
    if 'donemsel_sonuclar' in st.session_state:
        donemsel_sonuclar = st.session_state.donemsel_sonuclar
        p1 = donemsel_sonuclar['Periyot 1']
        p2 = donemsel_sonuclar['Periyot 2']
        
        st.subheader("Performans Metrikleri Karşılaştırması")
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Toplam Ciro", f"{p2['Toplam Ciro']:,.0f} €", f"{p2['Toplam Ciro'] - p1['Toplam Ciro']:,.0f} €")
        kpi_cols[1].metric("Aktif Müşteri Sayısı", f"{p2['Aktif Müşteri Sayısı']}", f"{p2['Aktif Müşteri Sayısı'] - p1['Aktif Müşteri Sayısı']}")
        kpi_cols[2].metric("Ortalama Sepet Tutarı", f"{p2['Ortalama Sepet Tutarı']:,.2f} €", f"{(p2['Ortalama Sepet Tutarı'] - p1['Ortalama Sepet Tutarı']):,.2f} €")
        kpi_cols[3].metric("Yeni Müşteri Sayısı", f"{p2['Yeni Müşteri Sayısı']}", f"{p2['Yeni Müşteri Sayısı'] - p1['Yeni Müşteri Sayısı']}")

        st.markdown("---")
        st.subheader("Müşteri Değer Göçü Analizi (CLV'ye Göre)")
        
        deger_gocu_verisi = st.session_state.deger_gocu_verisi
        
        if deger_gocu_verisi.empty:
            st.warning("Değer göçü analizi için yeterli veri bulunamadı.")
        else:
            # --- SANKEY RENKLENDİRME BAŞLANGIÇ ---
            deger_siralama = ["Yeni Müşteri", "Yüksek Değerli", "Orta Değerli", "Düşük Değerli", "Pasif / Churn"]
            deger_renkleri = {
                "Yeni Müşteri": "#4CAF50",         # Yeşil
                "Yüksek Değerli": "#2196F3",        # Mavi
                "Orta Değerli": "#FFC107",         # Amber
                "Düşük Değerli": "#FF9800",        # Turuncu
                "Pasif / Churn": "#F44336",         # Kırmızı
                "Tek Değer Grubu": "#9E9E9E"       # Gri
            }
            
            tum_etiketler = pd.concat([deger_gocu_verisi['Onceki_Durum'], deger_gocu_verisi['Simdiki_Durum']]).unique()
            etiketler = sorted(tum_etiketler, key=lambda x: deger_siralama.index(x) if x in deger_siralama else len(deger_siralama))
            etiket_map = {etiket: i for i, etiket in enumerate(etiketler)}

            node_colors = [deger_renkleri.get(etiket, '#CCCCCC') for etiket in etiketler]
            link_colors = [deger_renkleri.get(row['Onceki_Durum'], '#A0A0A0') for _, row in deger_gocu_verisi.iterrows()]
            # --- SANKEY RENKLENDİRME BİTİŞ ---

            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15, 
                    thickness=20, 
                    line=dict(color="black", width=0.5), 
                    label=etiketler,
                    color=node_colors # Düğüm renkleri eklendi
                ),
                link=dict(
                    source=deger_gocu_verisi['Onceki_Durum'].map(etiket_map),
                    target=deger_gocu_verisi['Simdiki_Durum'].map(etiket_map),
                    value=deger_gocu_verisi['deger'],
                    color=link_colors # Bağlantı renkleri eklendi
                )
            )])
            fig_sankey.update_layout(title_text="İki Dönem Arası Müşteri Değer Akışı", font_size=12)
            st.plotly_chart(fig_sankey, use_container_width=True)
            st.info("""
            **Grafik Nasıl Yorumlanır?** Bu grafik, seçtiğiniz iki dönem arasında müşteri değer segmentlerinin nasıl değiştiğini gösterir.
            - **Yeni Müşteri:** İlk dönemde aktif olmayıp ikinci dönemde aktif olan müşteriler.
            - **Pasif / Churn:** İlk dönemde aktif olup ikinci dönemde olmayan müşteriler.
            - **Diğer Akışlar:** ("Yüksek Değerli" -> "Orta Değerli" gibi) müşterilerinizin değer segmentlerindeki artış veya azalışları gösterir.
            """)